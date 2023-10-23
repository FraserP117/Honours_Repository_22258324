
# based on this implementation:
# __author__ = "Otto van der Himst"
# __credits__ = "Otto van der Himst, Pablo Lanillos"
# __version__ = "1.0"
# __email__ = "o.vanderhimst@student.ru.nl"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import datetime
import sys
# import gym
import gymnasium as gym
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file, version):
    # Calculate the moving average and standard deviation
    running_avg = np.zeros(len(scores))
    stds = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
        stds[i] = np.std(scores[max(0, i - 100):(i + 1)])
    
    # Create a figure and plot the moving average
    plt.figure(figsize=(12, 6))
    plt.plot(x, running_avg, label='100 episode MAVG of scores', color='b')
    
    # Plot the standard deviation bands
    plt.fill_between(x, running_avg - stds, running_avg + stds, color='b', alpha=0.3, label='Standard Deviation')
    
    # plt.title(f"100 episode MAVG of scores with Standard Deviation: {version}")
    plt.xlabel('Episodes', fontsize = 28)
    plt.ylabel('Score', fontsize = 28)
    # plt.legend()
    
    # Save or display the plot
    plt.show()
    plt.savefig(figure_file)

class ReplayMemory():

    # def __init__(self, capacity, obs_shape, device='cpu'):
    def __init__(self, capacity, obs_shape, device='cuda:0'):

        self.device=device

        self.capacity = capacity # The maximum number of items to be stored in memory

        # Initialize (empty) memory tensors
        self.obs_mem = torch.empty([capacity]+[dim for dim in obs_shape], dtype=torch.float32, device=self.device)
        self.action_mem = torch.empty(capacity, dtype=torch.int64, device=self.device)
        self.reward_mem = torch.empty(capacity, dtype=torch.int8, device=self.device)
        self.terminated_mem = torch.empty(capacity, dtype=torch.int8, device=self.device) # ADDED CODE
        self.truncated_mem = torch.empty(capacity, dtype=torch.int8, device=self.device) # ADDED CODE

        self.push_count = 0 # The number of times new data has been pushed to memory

    def push(self, obs, action, reward, terminated, truncated):

        # Store data to memory
        self.obs_mem[self.position()] = obs
        self.action_mem[self.position()] = action
        self.reward_mem[self.position()] = reward
        self.terminated_mem[self.position()] = terminated # ADDED CODE
        self.truncated_mem[self.position()] = truncated # ADDED CODE

        self.push_count += 1

    def position(self):
        # Returns the next position (index) to which data is pushed
        return self.push_count % self.capacity

    def sample(self, obs_indices, action_indices, reward_indices, terminated_indicies, truncated_indicies, max_n_indices, batch_size):
        # Fine as long as max_n is not greater than the fewest number of time steps an episode can take

        # Pick indices at random
        end_indices = np.random.choice(min(self.push_count, self.capacity)-max_n_indices*2, batch_size, replace=False) + max_n_indices

        # Correct for sampling near the position where data was last pushed
        for i in range(len(end_indices)):
            if end_indices[i] in range(self.position(), self.position()+max_n_indices):
                end_indices[i] += max_n_indices

        # Retrieve the specified indices that come before the end_indices
        obs_batch = self.obs_mem[np.array([index-obs_indices for index in end_indices])]
        action_batch = self.action_mem[np.array([index-action_indices for index in end_indices])]
        reward_batch = self.reward_mem[np.array([index-reward_indices for index in end_indices])]
        # done_batch = self.done_mem[np.array([index-done_indices for index in end_indices])] # OG CODE
        terminated_batch = self.terminated_mem[np.array([index-terminated_indicies for index in end_indices])]
        truncated_batch = self.truncated_mem[np.array([index-truncated_indicies for index in end_indices])]

        # Correct for sampling over multiple episodes
        for i in range(len(end_indices)):
            index = end_indices[i]
            for j in range(1, max_n_indices):
                if self.terminated_mem[index-j] or self.truncated_mem[index-j]: # if self.done_mem[index-j]: # OG CODE
                    for k in range(len(obs_indices)):
                        if obs_indices[k] >= j:
                            obs_batch[i, k] = torch.zeros_like(self.obs_mem[0])
                    for k in range(len(action_indices)):
                        if action_indices[k] >= j:
                            action_batch[i, k] = torch.zeros_like(self.action_mem[0]) # Assigning action '0' might not be the best solution, perhaps as assigning at random, or adding an action for this specific case would be better
                    for k in range(len(reward_indices)):
                        if reward_indices[k] >= j:
                            reward_batch[i, k] = torch.zeros_like(self.reward_mem[0]) # Reward of 0 will probably not make sense for every environment
                    for k in range(len(terminated_indicies)):
                        if terminated_indicies[k] >= j:
                            terminated_batch[i, k] = torch.zeros_like(self.terminated_mem[0])
                    for k in range(len(truncated_indicies)):
                        if truncated_indicies[k] >= j:
                            truncated_batch[i, k] = torch.zeros_like(self.truncated_mem[0])
                    break

        # return obs_batch, action_batch, reward_batch, done_batch # OG CODE
        return obs_batch, action_batch, reward_batch, terminated_batch, truncated_batch
    

class Model(nn.Module):
    
    def __init__(self, n_inputs, n_outputs, n_hidden=64, lr=1e-3, softmax=False, device='cuda:0'):
        super(Model, self).__init__()
        
        self.n_inputs = n_inputs # Number of inputs
        self.n_hidden = n_hidden # Number of hidden units
        self.n_outputs = n_outputs # Number of outputs
        self.softmax = softmax # If true apply a softmax function to the output
        
        self.fc1 = nn.Linear(self.n_inputs, self.n_hidden) # Hidden layer
        self.fc2 = nn.Linear(self.n_hidden, self.n_outputs) # Output layer
        
        self.optimizer = optim.Adam(self.parameters(), lr) # Adam optimizer
        
        self.device = device
        self.to(self.device)
    
    def forward(self, x):
        # Define the forward pass:
        h_relu = F.relu(self.fc1(x))
        y = self.fc2(h_relu)
        
        if self.softmax: # If true apply a softmax function to the output
            y = F.softmax(self.fc2(h_relu), dim=-1).clamp(min=1e-9, max=1-1e-9)
        
        return y

class MVGaussianModel(nn.Module):

    def __init__(self, n_inputs, n_outputs, n_hidden, lr=1e-3, dropout_prob=0.0, device='cuda:0', model=None, softmax = False):

        super(MVGaussianModel, self).__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.softmax = softmax # If true apply a softmax function to the output

        self.model = model

        # ADDED dropout layers with the specified dropout probability
        self.dropout1 = nn.Dropout(p = dropout_prob)
        self.dropout2 = nn.Dropout(p = dropout_prob)

        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)

        self.mean_fc = nn.Linear(n_hidden, n_outputs)
        self.stdev = nn.Linear(n_hidden, n_outputs)

        self.optimizer = optim.Adam(self.parameters(), lr)
        self.device = device
        self.to(self.device)

    def forward(self, x):

        x_1 = torch.relu(self.fc1(x))
        
        x_1 = self.dropout1(x_1)

        x_2 = torch.relu(self.fc2(x_1))
        
        x_2 = self.dropout2(x_2)

        if self.softmax: # If true apply a softmax function to the output
            y = F.softmax(self.fc2(x_1), dim=-1).clamp(min=1e-9, max=1-1e-9)

            return y

        mean = self.mean_fc(x_2)
        var = self.stdev(x_2) ** 2

        return mean, var

    def reparameterize(self, mean, var):
        std = torch.sqrt(var)
        epsilon = torch.randn_like(std)  # Sample from standard Gaussian
        sampled_value = mean + epsilon * std  # Reparameterization trick

        return sampled_value
    
class Agent():
    
    def __init__(self, argv):
        
        self.set_parameters(argv) # Set parameters
        
        self.obs_shape = self.env.observation_space.shape # The shape of observations
        self.obs_size = np.prod(self.obs_shape) # The size of the observation
        self.n_actions = self.env.action_space.n # The number of actions available to the agent
        
        self.freeze_cntr = 0 # Keeps track of when to (un)freeze the target network
        
        # # Initialize the networks:
        # self.transition_net = Model(self.obs_size+1, self.obs_size, self.n_hidden_trans, lr=self.lr_trans, device=self.device)
        # self.policy_net = Model(self.obs_size, self.n_actions, self.n_hidden_pol, lr=self.lr_pol, softmax=True, device=self.device)
        # self.value_net = Model(self.obs_size, self.n_actions, self.n_hidden_val, lr=self.lr_val, device=self.device)

        # Initialize the networks:
        # self.transition_net = MVGaussianModel(self.obs_size+1, self.obs_size, self.n_hidden_trans, lr=self.lr_trans, device=self.device)

        self.generative_transition_net = MVGaussianModel(self.obs_size+1, self.obs_size, self.n_hidden_gen_trans, lr=self.lr_gen_trans, device=self.device)
        self.variational_transition_net = MVGaussianModel(1, self.obs_size, self.n_hidden_var_trans, lr=self.lr_var_trans, device=self.device)
        self.generative_observation_net = MVGaussianModel(self.obs_size, self.obs_size, self.n_hidden_gen_obs, lr=self.lr_gen_obs, device=self.device, model = 'gen_obs')
        self.policy_net = Model(self.obs_size, self.n_actions, self.n_hidden_pol, lr=self.lr_pol, softmax=True, device=self.device)
        self.value_net = Model(self.obs_size, self.n_actions, self.n_hidden_val, lr=self.lr_val, device=self.device)
        
        if self.load_network: # If true: load the networks given paths

            # self.transition_net.load_state_dict(torch.load(self.network_load_path.format("trans")))
            # self.transition_net.eval()

            self.generative_transition_net.load_state_dict(torch.load(self.network_load_path.format("gen_trans")))
            self.generative_transition_net.eval()

            self.generative_observation_net.load_state_dict(torch.load(self.network_load_path.format("obs")))
            self.generative_observation_net.eval()

            self.variational_transition_net.load_state_dict(torch.load(self.network_load_path.format("var_trans")))
            self.variational_transition_net.eval()

            self.policy_net.load_state_dict(torch.load(self.network_load_path.format("pol")))
            self.policy_net.eval()

            self.value_net.load_state_dict(torch.load(self.network_load_path.format("val")))
            self.value_net.eval()

        self.target_net = Model(self.obs_size, self.n_actions, self.n_hidden_val, lr=self.lr_val, device=self.device)
        self.target_net.load_state_dict(self.value_net.state_dict())
        
        # Initialize the replay memory
        self.memory = ReplayMemory(self.memory_capacity, self.obs_shape, device=self.device)
        
        # When sampling from memory at index i, obs_indices indicates that we want observations with indices i-obs_indices, works the same for the others
        # self.obs_indices = [2, 1, 0]
        self.obs_indices = [(self.n_screens+1)-i for i in range(self.n_screens+2)]
        self.action_indices = [2, 1]
        self.reward_indices = [1]

        # self.done_indices = [0]
        # self.max_n_indices = max(max(self.obs_indices, self.action_indices, self.reward_indices, self.done_indices)) + 1

        self.terminated_indicies = [0]
        self.truncated_indicies = [0]

        self.max_n_indices = max(max(self.obs_indices, self.action_indices, self.reward_indices, self.terminated_indicies, self.truncated_indicies)) + 1


    def set_parameters(self, argv):
        
        # The default parameters
        default_parameters = {'run_id':"_rX", 'device':'cpu',
              'env':'CartPole-v1', 'n_episodes':2000, 'n_screens':4,
              'n_hidden_var_trans':64, 'lr_var_trans':1e-4,
              'n_hidden_trans':64, 'lr_trans':1e-3, 
              'n_hidden_gen_obs':64, 'lr_gen_obs': 1e-4,
              'n_hidden_gen_trans':64, 'lr_gen_trans':1e-4,
              'n_hidden_pol':64, 'lr_pol':1e-3, 
              'n_hidden_val':64, 'lr_val':1e-4,
              'memory_capacity':65536, 'batch_size':64, 'freeze_period':25,
              'Beta':0.99, 'gamma':1.00, 
              'print_timer':100,
              'keep_log':True, 'log_path':"logs/ai_mdp_log{}.txt", 'log_save_timer':10,
              'save_results':True, 'results_path':"results/ai_mdp_results{}.npz", 'results_save_timer':500,
              'save_network':True, 'network_save_path':"networks/ai_mdp_{}net{}.pth", 'network_save_timer':500,
              'load_network':False, 'network_load_path':"networks/ai_mdp_{}net_rX.pth"}
        # Possible command:
            # python ai_mdp_agent.py device=cuda:0
        
        # Adjust the custom parameters according to the arguments in argv
        custom_parameters = default_parameters.copy()
        custom_parameter_msg = "Custom parameters:\n"
        for arg in argv:
            key, value = arg.split('=')
            if key in custom_parameters:
                custom_parameters[key] = value
                custom_parameter_msg += "  {}={}\n".format(key, value)
            else:
                print("Argument {} is unknown, terminating.".format(arg))
                sys.exit()
        
        def interpret_boolean(param):
            if type(param) == bool:
                return param
            elif param in ['True', '1']:
                return True
            elif param in ['False', '0']:
                return False
            else:
                sys.exit("param '{}' cannot be interpreted as boolean".format(param))
        
        # Set all parameters
        self.run_id = custom_parameters['run_id'] # Is appended to paths to distinguish between runs
        self.device = custom_parameters['device'] # The device used to run the code

        self.n_screens = int(custom_parameters['n_screens']) # The number of obervations (screens) that are passed to the VAE
        
        # self.env = gym.make(custom_parameters['env']) # The environment in which to train # OG
        self.env = gym.make(custom_parameters['env'], render_mode = "human") # The environment in which to train
        self.n_episodes = int(custom_parameters['n_episodes']) # The number of episodes for which to train
        
        # Set number of hidden nodes and learning rate for each network
        self.n_hidden_gen_trans = int(custom_parameters['n_hidden_gen_trans'])
        self.lr_gen_trans = float(custom_parameters['lr_gen_trans'])

        self.n_hidden_trans = int(custom_parameters['n_hidden_trans'])
        self.lr_trans = float(custom_parameters['lr_trans'])
        self.n_hidden_pol = int(custom_parameters['n_hidden_pol'])
        self.lr_pol = float(custom_parameters['lr_pol'])

        self.n_hidden_gen_obs = int(custom_parameters['n_hidden_gen_obs'])
        self.lr_gen_obs = float(custom_parameters['lr_gen_obs'])

        self.n_hidden_var_trans = int(custom_parameters['n_hidden_var_trans'])
        self.lr_var_trans = float(custom_parameters['lr_var_trans'])

        self.n_hidden_val = int(custom_parameters['n_hidden_val'])
        self.lr_val = float(custom_parameters['lr_val'])
        
        self.memory_capacity = int(custom_parameters['memory_capacity']) # The maximum number of items to be stored in memory
        self.batch_size = int(custom_parameters['batch_size']) # The mini-batch size
        self.freeze_period = int(custom_parameters['freeze_period']) # The number of time-steps the target network is frozen
        
        self.gamma = float(custom_parameters['gamma']) # A precision parameter
        self.Beta = float(custom_parameters['Beta']) # The discount rate
        
        self.print_timer = int(custom_parameters['print_timer']) # Print progress every print_timer episodes
        
        self.keep_log = interpret_boolean(custom_parameters['keep_log']) # If true keeps a (.txt) log concerning data of this run
        self.log_path = custom_parameters['log_path'].format(self.run_id) # The path to which the log is saved
        self.log_save_timer = int(custom_parameters['log_save_timer']) # The number of episodes after which the log is saved
        
        self.save_results = interpret_boolean(custom_parameters['save_results']) # If true saves the results to an .npz file
        self.results_path = custom_parameters['results_path'].format(self.run_id) # The path to which the results are saved
        self.results_save_timer = int(custom_parameters['results_save_timer']) # The number of episodes after which the results are saved
        
        self.save_network = interpret_boolean(custom_parameters['save_network']) # If true saves the policy network (state_dict) to a .pth file
        self.network_save_path = custom_parameters['network_save_path'].format("{}", self.run_id) # The path to which the network is saved
        self.network_save_timer = int(custom_parameters['network_save_timer']) # The number of episodes after which the network is saved
                
        self.load_network = interpret_boolean(custom_parameters['load_network']) # If true loads a (policy) network (state_dict) instead of initializing a new one
        self.network_load_path = custom_parameters['network_load_path'] # The path from which to laod the network
        
        msg = "Default parameters:\n"+str(default_parameters)+"\n"+custom_parameter_msg
        print(msg)
        
        if self.keep_log: # If true: write a message to the log
            self.record = open(self.log_path, "a")
            self.record.write("\n\n-----------------------------------------------------------------\n")
            self.record.write("File opened at {}\n".format(datetime.datetime.now()))
            self.record.write(msg+"\n")
        
    def select_action(self, obs):
        with torch.no_grad():

            # Determine the action distribution given the current observation:
            policy = self.policy_net(obs)

            # print(f"policy: {policy}")

            return torch.multinomial(policy, 1)

    def kl_divergence_diag_cov_gaussian(self, mu1, sigma1_sq, mu2, sigma2_sq): 
        '''
        Returns the average KL divergence for batched inputs. 
        In case of scaler inputs, simply returns the KL divergence for the two input distributions

        mu1: 
            a batch of predicted mean vectors for the first distribution, 
            either contains a single entry or is a batch of 64 mean vectors
        mu2: 
            a batch of predicted mean vectors for the second distribution, 
            either contains a single entry or is a batch of 64 mean vectors
        sigma1_sq: 
            a batch of predicted variances for the first distribution, 
            either contains a single entry or is a batch of 64 mean vectors
        sigma2_sq:
            a batch of predicted variances for the second distribution, 
            either contains a single entry or is a batch of 64 mean vectors
        '''

        avg_kl_div = 0

        for i in range(len(mu1)):

            kl_div_i = 0.5 * torch.sum(
                (torch.sqrt(sigma2_sq[i]) / torch.sqrt(sigma1_sq[i])) + ((mu1[i] - mu2[i])**2 / sigma1_sq[i]) - 1 + torch.log(torch.sqrt(sigma1_sq[i]) / torch.sqrt(sigma2_sq[i]))
            )

            avg_kl_div += kl_div_i

        avg_kl_div = avg_kl_div / len(mu1)

        return avg_kl_div

    def get_mini_batches(self):

        # Retrieve transition data in mini batches
        all_obs_batch, all_actions_batch, reward_batch_t1, terminated_batch_t2, truncated_batch_t2 = self.memory.sample(
                self.obs_indices, self.action_indices, self.reward_indices,
                self.terminated_indicies, self.truncated_indicies, self.max_n_indices, self.batch_size)
        
        # Retrieve a batch of observations for 3 consecutive points in time
        obs_batch_t0 = all_obs_batch[:, 0].view([self.batch_size] + [dim for dim in self.obs_shape])
        obs_batch_t1 = all_obs_batch[:, 1].view([self.batch_size] + [dim for dim in self.obs_shape])
        obs_batch_t2 = all_obs_batch[:, 2].view([self.batch_size] + [dim for dim in self.obs_shape])
        
        # Retrieve the agent's action history for time t0 and time t1
        action_batch_t0 = all_actions_batch[:, 0].unsqueeze(1)
        action_batch_t1 = all_actions_batch[:, 1].unsqueeze(1)
        
        # At time t0 predict the state at time t1:
        X_0 = torch.cat((obs_batch_t0, action_batch_t0.float()), dim=1)
        X_1 = torch.cat((obs_batch_t1, action_batch_t1.float()), dim=1)

        pred_mean_state_t0t1_theta, pred_var_state_t0t1_theta = self.generative_transition_net(X_0)
        # pred_mean_state_t1t2_theta, pred_var_state_t1t2_theta = self.generative_transition_net(X_1)

        pred_batch_mean_state_t0t1_phi, pred_batch_var_state_t0t1_phi = self.variational_transition_net(action_batch_t0.float())

        pred_batch_mean_obs_t0t1_phi, pred_batch_var_obs_t0t1_phi = self.generative_observation_net(pred_batch_mean_state_t0t1_phi)

        D_KL_state_pred_batch_theta_phi = self.kl_divergence_diag_cov_gaussian(
             pred_mean_state_t0t1_theta, pred_var_state_t0t1_theta,
             pred_batch_mean_state_t0t1_phi, pred_batch_var_state_t0t1_phi
        )

        return (
            obs_batch_t0,
            obs_batch_t1,
            obs_batch_t2,

            pred_mean_state_t0t1_theta, pred_var_state_t0t1_theta,
            pred_batch_mean_state_t0t1_phi, pred_batch_var_state_t0t1_phi,
            pred_batch_mean_obs_t0t1_phi, pred_batch_var_obs_t0t1_phi,

            D_KL_state_pred_batch_theta_phi,

            action_batch_t0, action_batch_t1,

            reward_batch_t1,

            terminated_batch_t2, truncated_batch_t2
        )

    def compute_value_net_loss(
        self, 
        obs_batch_t1,
        obs_batch_t2,

        action_batch_t1, reward_batch_t1,
        terminated_batch_t2, truncated_batch_t2, D_KL_state_pred_batch_theta_phi
    ):

        with torch.no_grad():

            # Determine the action distribution for time t2:
            policy_batch_t2 = self.policy_net(obs_batch_t2)
            # print(f"policy_batch_t2: {policy_batch_t2}")

            # Determine the target EFEs for time t2:
            target_EFEs_batch_t2 = self.target_net(obs_batch_t2)

            # Weigh the target EFEs according to the action distribution:
            weighted_EFE_value_targets = ((1-(terminated_batch_t2 | truncated_batch_t2)) * policy_batch_t2 *
                                target_EFEs_batch_t2).sum(-1).unsqueeze(1)

            # # Determine the batch of bootstrapped estimates of the EFEs:
            EFE_estimate_batch = -reward_batch_t1 + D_KL_state_pred_batch_theta_phi + self.Beta * weighted_EFE_value_targets

        # Determine the EFE at time t1 according to the value network:
        EFE_batch_t1 = self.value_net(obs_batch_t1).gather(1, action_batch_t1)

        # Determine the MSE loss between the EFE estimates and the value network output:
        EFE_efe_value_net_loss = F.mse_loss(EFE_estimate_batch, EFE_batch_t1)

        return EFE_efe_value_net_loss

    def compute_VFE(self, obs_batch_t1, VFE_state_pred_batch):
        
        # Determine the action distribution for time t1:
        policy_batch_t1 = self.policy_net(obs_batch_t1)
        # print(f"policy_batch_t1: {policy_batch_t1}")
        
        # Determine the EFEs for time t1:
        EFEs_batch_t1 = self.value_net(obs_batch_t1).detach() 

        # Take a gamma-weighted Boltzmann distribution over the EFEs:
        boltzmann_EFEs_batch_t1 = torch.softmax(-self.gamma * EFEs_batch_t1, dim=1).clamp(min=1e-9, max=1-1e-9) # shape: [64, 2]

        
        # Weigh them according to the action distribution:
        energy_policy_pred_batch = -(policy_batch_t1 * torch.log(boltzmann_EFEs_batch_t1)).sum(-1).view(self.batch_size, 1) # IS NAN
        
        # Determine the entropy of the action distribution
        entropy_policy_pred_batch = -(policy_batch_t1 * torch.log(policy_batch_t1)).sum(-1).view(self.batch_size, 1)
        
        # Determine the VFE, then take the mean over all batch samples:
        VFE_batch = VFE_state_pred_batch + (energy_policy_pred_batch - entropy_policy_pred_batch)


        VFE = torch.mean(VFE_batch)
                
        return VFE

    def negative_log_likelihood(self, observation_samples, predicted_cur_obs_mean, predicted_cur_obs_var):
        '''
        Intended to calculate the average negative log likelihood across a batch of 64 inputs.

        observation_samples: a batch of samples drawn from the observation distribution by means of the reparameterisation trick
        predicted_cur_obs_mean: a batch of predicted mean vectors for the observation distribution
        predicted_cur_obs_var: a batch of diagonal elements of the covariace matrix for the observation distribution. 

        batch size is 64.
        the observation distribution is assumed to be a diagonal multivariate Gaussian distribution. 

        Perhaps it's better to average over predicted_cur_obs_mean and average over predicted_cur_obs_var?        '''

        # Calculate dimensionality (d) from the length of predicted_cur_obs_var
        d = len(observation_samples[0])

        # init the average negatice log likelihood value:
        avg_neg_log_li = 0

        # iterate across the batch of samples:
        for i in range(len(observation_samples)):

            observation_samples[i] = observation_samples[i].to(self.device)
            predicted_cur_obs_mean[i] = predicted_cur_obs_mean[i].to(self.device)
            predicted_cur_obs_var[i] = predicted_cur_obs_var[i].to(self.device)

            # Create a diagonal covariance matrix Sigma
            Sigma_i = torch.diag(predicted_cur_obs_var[i])

            # Calculate the inverse of Sigma
            Sigma_i_inv = torch.inverse(Sigma_i)

            # Calculate the difference between the ith observation sample and the ith predicted mean vector:
            observation_samples_minus_predicted_cur_obs_mean = observation_samples[i] - predicted_cur_obs_mean[i]

            # Calculate the Mahalanobis distance term
            quad_term = torch.matmul(
                torch.matmul(
                    observation_samples_minus_predicted_cur_obs_mean.unsqueeze(0), 
                    Sigma_i_inv
                ).to(self.device), 
                observation_samples_minus_predicted_cur_obs_mean.unsqueeze(1)
            ).to(self.device)

            # Calculate the log determinant term
            log_det_term = 0.5 * torch.log(torch.det(Sigma_i)).to(self.device)

            # Calculate the constant term
            constant_term = 0.5 * d * torch.log(2 * torch.tensor([np.pi], dtype=torch.float32)).to(self.device)

            # Calculate the negative log likelihood (surprisal)
            S_observation_samples = quad_term + log_det_term + constant_term

            # Sum all the elements to get a scalar result
            # S_observation_samples_scalar = S_observation_samples.sum()
            S_observation_samples_scalar = S_observation_samples.item()

            # avg_neg_log_li += S_observation_samples_scalar.item()
            avg_neg_log_li += S_observation_samples_scalar


        # return the average neg_log_li across the batch of 64 samples:
        avg_neg_log_li_final = avg_neg_log_li/len(observation_samples)

        return avg_neg_log_li_final



    def compute_VFE_pomdp(
        self, 
        pred_batch_mean_state_t0t1_phi, pred_batch_var_state_t0t1_phi,
        pred_batch_mean_obs_t0t1_phi, pred_batch_var_obs_t0t1_phi,
        reparamed_observation_samples, 
        D_KL_state_pred_batch_theta_phi
    ):
        
        # Determine the action distribution for time t1:
        policy_batch_t1 = self.policy_net(pred_batch_mean_state_t0t1_phi)

        
        # Determine the EFEs for time t1:
        EFEs_batch_t1 = self.value_net(pred_batch_mean_state_t0t1_phi)

        # Take a gamma-weighted Boltzmann distribution over the EFEs:
        boltzmann_EFEs_batch_t1 = torch.softmax(-self.gamma * EFEs_batch_t1, dim=1).clamp(min=1e-9, max=1-1e-9)
        
        # Weigh them according to the action distribution:
        energy_term_batch = -(policy_batch_t1 * torch.log(boltzmann_EFEs_batch_t1)).sum(-1).unsqueeze(1)
        
        # Determine the entropy of the action distribution
        entropy_batch = -(policy_batch_t1 * torch.log(policy_batch_t1)).sum(-1).unsqueeze(1)

        neg_log_li = self.negative_log_likelihood(
            predicted_cur_obs_mean = pred_batch_mean_obs_t0t1_phi, 
            predicted_cur_obs_var = pred_batch_var_obs_t0t1_phi, 
            observation_samples = reparamed_observation_samples
        )
        
        # Determine the VFE, then take the mean over all batch samples:
        VFE_batch = neg_log_li + D_KL_state_pred_batch_theta_phi + (energy_term_batch - entropy_batch)

        VFE = torch.mean(VFE_batch)
        
        return VFE

    def compute_entropy(probabilities):
        # Calculate entropy given probabilities
        log_probabilities = torch.log(probabilities)
        entropy = -torch.sum(probabilities * log_probabilities, dim=-1)
        return entropy


    def learn(self):


        # If there are not enough transitions stored in memory, return:
        if self.memory.push_count - self.max_n_indices*2 < self.batch_size:
            return

        # After every freeze_period time steps, update the target network:
        if self.freeze_cntr % self.freeze_period == 0:
            self.target_net.load_state_dict(self.value_net.state_dict())
        self.freeze_cntr += 1


        (
            obs_batch_t0,
            obs_batch_t1,
            obs_batch_t2,

            pred_mean_state_t0t1_theta, pred_var_state_t0t1_theta,
            pred_batch_mean_state_t0t1_phi, pred_batch_var_state_t0t1_phi,
            pred_batch_mean_obs_t0t1_phi, pred_batch_var_obs_t0t1_phi,

            D_KL_state_pred_batch_theta_phi,

            action_batch_t0, action_batch_t1,

            reward_batch_t1,

            terminated_batch_t2, truncated_batch_t2
        ) = self.get_mini_batches()


        efe_value_net_loss = self.compute_value_net_loss(
            obs_batch_t1,
            obs_batch_t2,

            action_batch_t1, reward_batch_t1,
            terminated_batch_t2, truncated_batch_t2, D_KL_state_pred_batch_theta_phi
        )

        reparamed_observation_samples = self.generative_observation_net.reparameterize(pred_batch_mean_obs_t0t1_phi, pred_batch_var_obs_t0t1_phi)


        VFE = self.compute_VFE_pomdp(
            pred_batch_mean_state_t0t1_phi, pred_batch_var_state_t0t1_phi,
            pred_batch_mean_obs_t0t1_phi, pred_batch_var_obs_t0t1_phi,
            reparamed_observation_samples, 
            D_KL_state_pred_batch_theta_phi
        )

        # Reset the gradients:
        self.generative_transition_net.optimizer.zero_grad()
        self.generative_observation_net.optimizer.zero_grad()
        self.variational_transition_net.optimizer.zero_grad()
        self.policy_net.optimizer.zero_grad()
        self.value_net.optimizer.zero_grad()

        # Compute the gradients:
        VFE.backward()
        efe_value_net_loss.backward()

        # Perform gradient descent:
        self.generative_transition_net.optimizer.step()
        self.generative_observation_net.optimizer.step()
        self.variational_transition_net.optimizer.step()
        self.policy_net.optimizer.step()
        self.value_net.optimizer.step()

        return VFE
        
    def train(self):

        filename = f"Deep_AIF_MDP_Cart_Pole_v1"
        figure_file = f"plots/{filename}.png"

        msg = "Environment is: {}\nTraining started at {}".format(self.env.unwrapped.spec.id, datetime.datetime.now())
        print(msg)
        if self.keep_log:
            self.record.write(msg+"\n")

        # noise_std = 0.1
        noise_std = 0
        
        results = []
        for ith_episode in range(self.n_episodes):
            
            total_reward = 0
            obs, info = self.env.reset()

            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            obs = obs + noise_std * np.random.randn(*obs.shape)
            obs = torch.tensor(obs, dtype = torch.float32, device = self.device)

            terminated = False
            truncated = False
            done = terminated or truncated

            reward = 0
            while not done:
                
                action = self.select_action(obs)
                
                # self.memory.push(obs, action, reward, done)
                self.memory.push(obs, action, reward, terminated, truncated)
                
                # obs, reward, done, _ = self.env.step(action[0].item())
                obs, reward, terminated, truncated, _  = self.env.step(action[0].item())
                # obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                obs = obs + noise_std * np.random.randn(*obs.shape)
                obs = torch.tensor(obs, dtype = torch.float32, device = self.device)

                total_reward += reward
                
                VFE = self.learn()
                
                # if done:
                #     self.memory.push(obs, -99, -99, done)

                print(f"\nselected action: {action}")
                print(f"obs: {obs}")
                print(f"free_energy_loss: {VFE}")
                print(f"total_reward: {total_reward}")
                print(f"ith_episode: {ith_episode}/{self.n_episodes}\n")

                # breakpoint()

                if terminated or truncated:
                    done = True
                    self.memory.push(obs, -99, -99, terminated, truncated)

            results.append(total_reward)
            
            # Print and keep a (.txt) record of stuff
            if ith_episode > 0 and ith_episode % self.print_timer == 0:
                avg_reward = np.mean(results)
                last_x = np.mean(results[-self.print_timer:])
                msg = "Episodes: {:4d}, avg score: {:3.2f}, over last {:d}: {:3.2f}".format(ith_episode, avg_reward, self.print_timer, last_x)
                print(msg)
                
                if self.keep_log:
                    self.record.write(msg+"\n")
                    
                    if ith_episode % self.log_save_timer == 0:
                        self.record.close()
                        self.record = open(self.log_path, "a")
            
            # If enabled, save the results and the network (state_dict)
            if self.save_results and ith_episode > 0 and ith_episode % self.results_save_timer == 0:
                np.savez("results/intermediary/intermediary_results{}_{:d}".format(self.run_id, ith_episode), np.array(results))
            if self.save_network and ith_episode > 0 and ith_episode % self.network_save_timer == 0:

                # torch.save(self.transition_net.state_dict(), "networks/intermediary/intermediary_transnet{}_{:d}.pth".format(self.run_id, ith_episode))

                torch.save(self.generative_transition_net.state_dict(), "networks/intermediary/intermediary_gen_transnet{}_{:d}.pth".format(self.run_id, ith_episode))
                torch.save(self.variational_transition_net.state_dict(), "networks/intermediary/intermediary_var_transnet{}_{:d}.pth".format(self.run_id, ith_episode))

                torch.save(self.policy_net.state_dict(), "networks/intermediary/intermediary_polnet{}_{:d}.pth".format(self.run_id, ith_episode))
                torch.save(self.value_net.state_dict(), "networks/intermediary/intermediary_valnet{}_{:d}.pth".format(self.run_id, ith_episode))
        
        self.env.close()
        
        # If enabled, save the results and the network (state_dict)
        if self.save_results:
            np.savez("results/intermediary/intermediary_results{}_end".format(self.run_id), np.array(results))
            np.savez(self.results_path, np.array(results))
        if self.save_network:

            # torch.save(self.transition_net.state_dict(), "networks/intermediary/intermediary_transnet{}_end.pth".format(self.run_id))

            torch.save(self.generative_transition_net.state_dict(), "networks/intermediary/intermediary_gen_transnet{}_end.pth".format(self.run_id))
            torch.save(self.variational_transition_net.state_dict(), "networks/intermediary/intermediary_var_transnet{}_end.pth".format(self.run_id))

            torch.save(self.policy_net.state_dict(), "networks/intermediary/intermediary_polnet{}_end.pth".format(self.run_id))
            torch.save(self.value_net.state_dict(), "networks/intermediary/intermediary_valnet{}_end.pth".format(self.run_id))
            
            # torch.save(self.transition_net.state_dict(), self.network_save_path.format("trans"))

            torch.save(self.generative_transition_net.state_dict(), self.network_save_path.format("gen_trans"))
            torch.save(self.variational_transition_net.state_dict(), self.network_save_path.format("var_trans"))

            torch.save(self.policy_net.state_dict(), self.network_save_path.format("pol"))
            torch.save(self.value_net.state_dict(), self.network_save_path.format("val"))
        
        # Print and keep a (.txt) record of stuff
        msg = "Training finished at {}".format(datetime.datetime.now())
        print(msg)
        if self.keep_log:
            self.record.write(msg)
            self.record.close()

        x = [i + 1 for i in range(self.n_episodes)]
        plot_learning_curve(x, results, figure_file, "AcT Action Selection")

        # plt.plot(x, VFE_Losses)
        # plt.show()
                
if __name__ == "__main__":
    agent = Agent(sys.argv[1:])
    agent.train()