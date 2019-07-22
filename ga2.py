#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from netsapi.challenge import *
import pandas as pd
import numpy as np
import itertools


# In[28]:


class RemyGA:
  '''
    Simple Genetic Algorithm. 
    https://github.com/slremy/estool/
  '''
  def __init__(self, num_params,      # number of model parameters
               random_individuals_fcn,
               mutate_fcn,
               immigrant_ratio=.2,     # percentage of new individuals
               sigma_init=0.1,        # initial standard deviation
               sigma_decay=0.999,     # anneal standard deviation
               sigma_limit=0.01,      # stop annealing if less than this
               popsize=16,           # population size
               elite_ratio=0.1,       # percentage of the elites
               forget_best=False,     # forget the historical best elites
               weight_decay=0.01,     # weight decay coefficient
              ):

    self.num_params = num_params
    self.sigma_init = sigma_init
    self.sigma_decay = sigma_decay
    self.sigma_limit = sigma_limit
    self.popsize = popsize
    self.random_individuals_fcn = random_individuals_fcn
    self.mutate_fcn = mutate_fcn
    self.solutions = None

    self.elite_ratio = elite_ratio
    self.elite_popsize = int(self.popsize * self.elite_ratio)
    self.immigrant_ratio = immigrant_ratio
    self.immigrant_popsize = int(self.popsize * self.immigrant_ratio)

    self.sigma = self.sigma_init
    self.elite_params = np.zeros((self.elite_popsize, self.num_params))
    #self.elite_rewards = np.zeros(self.elite_popsize)
    self.best_param = np.zeros(self.num_params)
    self.best_reward = 0
    self.reward_pdf = np.zeros(self.popsize+1)
    self.solutions = np.zeros((self.popsize, self.num_params))
    self.first_iteration = True
    self.forget_best = forget_best
    self.weight_decay = weight_decay

  def rms_stdev(self):
    return self.sigma # same sigma for all parameters.

  def ask(self, process=lambda x:x):
    '''returns a list of parameters'''
    self.epsilon = np.random.randn(self.popsize, self.num_params) * self.sigma
    
    
  
    
    def crossover(a,b):
      
      cross_point = int((self.num_params // 2)*np.random.rand(1)) * 2 - 1;
      c = np.append(a[:cross_point], b[cross_point:self.num_params]);
      return c
    
    index_array = np.arange(self.popsize)
    if self.first_iteration:
        self.solutions = process(self.random_individuals_fcn(self.popsize,self.num_params))
    else:
        #intialize the index list for "mating" chromosomes
        childrenIDX = range(self.popsize - self.elite_popsize - self.immigrant_popsize);
        selected = np.arange(2*len(childrenIDX));
        for i in range(len(selected)):
            testNo = 1;
            #Choose a parent
            while self.reward_pdf[testNo] < np.random.rand():
                testNo = testNo + 1;
            selected[i] = index_array[testNo];
        children = []
        for i in range(len(childrenIDX)):
            chromosomeA = self.solutions[selected[i*2+0], :];
            chromosomeB = self.solutions[selected[i*2+1], :];
            child = crossover(chromosomeA,chromosomeB) if 0.5 < np.random.rand() else crossover(chromosomeB,chromosomeA)
            children.append(self.mutate_fcn(child))
        
        self.solutions = process(np.concatenate((self.elite_params, self.random_individuals_fcn(self.immigrant_popsize,self.num_params), np.array(children))))

    return self.solutions

  def tell(self, reward_table_result):
    # input must be a numpy float array
    assert(len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."

    reward_table = np.array(reward_table_result)
    
    if self.weight_decay > 0:
      l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
      reward_table += l2_decay

    reward = reward_table
    solution = self.solutions

    reward_masked = np.ma.masked_array(reward,mask = (np.isnan(reward) | np.isinf(reward)))
    
    self.reward_pdf = (np.cumsum(reward_masked)/np.sum(reward_masked)).compressed()
    sorted_idx = np.argsort(reward_masked)[::-1]
    idx = sorted_idx[~reward_masked.mask][0:self.elite_popsize]
    
    assert(len(idx) == self.elite_popsize), "Inconsistent elite size reported."
    
    self.elite_rewards = reward[idx]
    self.elite_params = solution[idx]

    self.curr_best_reward = self.elite_rewards[0]
    
    if self.first_iteration or (self.curr_best_reward > self.best_reward):
      self.first_iteration = False
      self.best_reward = self.elite_rewards[0]
      self.best_param = np.copy(self.elite_params[0])

    if (self.sigma > self.sigma_limit):
      self.sigma *= self.sigma_decay

    self.first_iteration = False

  def current_param(self):
    return self.elite_params[0]

  def set_mu(self, mu):
    pass

  def best_param(self):
    return self.best_param

  def result(self): # return best params so far, along with historically best reward, curr reward, sigma
    return (self.best_param, self.best_reward, self.curr_best_reward, self.sigma)


def mutate(chromosome):
    mutation_rate = .5
    for j in range(chromosome.shape[0] // 2):
        left = j * 2
        right = j * 2 + 1
        r = np.random.rand(1);
        if(r > mutation_rate):
            mutetype = np.random.rand(1)
            if mutetype > 0.5:
                chromosome[left] = np.remainder(chromosome[left]+np.random.randn(1) * 0.4 ,0.99);
                chromosome[right] = np.remainder(chromosome[right]+np.random.randn(1) * 0.4 ,0.99);
            else:
                r2 = np.random.rand(1) 
                if ( r2 > 0.5):
                    chromosome[right] = 1 - chromosome[right]
                else:
                    chromosome[left] = 1 - chromosome[left]
    return chromosome


def make_random_individuals(x,y):
    value = np.random.choice(act_space , (x,y) )
    
    for i in range(x):
        for j in range(y // 2):
            if value[i][ j * 2] + value[i][ j * 2 + 1] > 1.4 or value[i][ j * 2] + value[i][ j * 2 + 1] <= 0.2:
                if np.random.rand(1) > 0.5:
                    value[i][j * 2 + 1] = 1 - value[i][j * 2]
                else:
                    value[i][j * 2 ] = 1 - value[i][j * 2 + 1]
    return value

def boundary(individual):
    processed = individual%(1+np.finfo(float).eps)
    return processed


# In[29]:



act_space = [ 0 , 0.2 , 0.4 , 0.6 , 0.8 , 1]


# In[30]:


class SRGAAgent():
    def __init__(self, environment):
        self._epsilon = 0.2  # 20% chances to apply a random action
        self._gamma = 0.99  # Discounting factor
        self._alpha = 0.5  # soft update param
        
        
        self.environment = environment #self._env = env
        
        self.popsize= 5
        self.num_paramters = 10
        self.solver = RemyGA(self.num_paramters,         # number of model parameters
                random_individuals_fcn=make_random_individuals,
                mutate_fcn = mutate,
                sigma_init=1,          # initial standard deviation
                popsize=self.popsize,       # population size
                elite_ratio=0.2,       # percentage of the elites
                forget_best=False,     # forget the historical best elites
                weight_decay=0.00,     # weight decay coefficient
                )
    def stateSpace(self):
        return range(1,self.environment.policyDimension+1)

    def train(self):
        allrewards = []
        for episode in range(20):
            rewards = []
            if episode % self.popsize == 0:
                # ask for a set of random candidate solutions to be evaluated
                solutions = self.solver.ask(boundary)
            
                #convert an array of 10 floats into a policy of itn, irs per year for 5 years
                policies = []
                for v in solutions:
                    actions = [i for i in itertools.zip_longest(*[iter(v)] * 2, fillvalue="")]
                    policy = {i+1: list(actions[i]) for i in range(5)}
                    policies.append(policy)

                # calculate the reward for each given solution using the environment's method
                batchRewards = self.environment.evaluatePolicy(policies)
                rewards.append(batchRewards)

                self.solver.tell(batchRewards);

                allrewards.append(rewards)
        return np.array(allrewards)

    def generate(self):
        self.train()
        #generate a policy from the array used to represent the candidate solution
        actions = [i for i in itertools.zip_longest(*[iter(self.solver.best_param)] * 2, fillvalue="")]
        best_policy = {state: list(actions[state-1]) for state in self.stateSpace()}
        best_reward = self.environment.evaluatePolicy(best_policy)
        print(best_policy, best_reward)
        return best_policy, best_reward


# In[31]:


EvaluateChallengeSubmission(ChallengeProveEnvironment , SRGAAgent, "SRGAAgent_20.csv")


# In[26]:


test = pd.read_csv("SRGAAgent_20.csv") 
print(test)


# In[ ]:





# In[ ]:




