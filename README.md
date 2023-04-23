# RL_18

## Assignement 21.01


### 1.0 Explanation of the Actor Critic Algorythm

""""
 Explanation and documentation of the Actor-Critic algorithm

    The Actor-Critic algorithm combines the policy-gradient methods 
with the value-function methods. It comprises two main components: the Actor and 
the Critic. The Actor is responsible for selecting the best action given the 
current state, and the Critic evaluates the value function of the state. 
The Critic helps the Actor to learn better policies by providing an estimate 
of the value function.

    In the provided code, the ActorCritic class is defined 
as a subclass of the nn.Module class. The class has an actor and a critic,
which are instances of the MLP class (Multilayer Perceptron). 
The forward method returns the probability distribution over actions
and the value of the input state. The get_action method is responsible
for sampling an action from the probability distribution and returning 
the action, its log probability, and the value.

    The update_ac method is responsible for updating the Actor 
and Critic networks. It takes in the following parameters:

    - network: The Actor-Critic network.
    - rewards: A list of rewards from the environment.
    - log_probs: A list of log probabilities of the chosen actions.
    - values: A list of values of the states visited.
    - masks: A list of masks indicating whether an episode has terminated.
    - Qval: The estimated value of the next state.
    - gamma: The discount factor.

    The method computes the Q values by calling the calculate_returns function.
It then calculates the advantage, which is the difference between the Q values 
and the state values. The Actor loss is computed as the mean of the element-wise 
product of the negative log probabilities and the detached advantages. 
The Critic loss is computed as the mean of the squared advantages multiplied by 0.5. 
The overall loss, ac_loss, is the sum of the Actor loss and the Critic loss.

Entropy regularization term:

    Entropy regularization is used to encourage exploration by penalizing 
low-entropy policies. This means the agent will be less likely to get stuck 
in a suboptimal deterministic policy. To include the entropy regularization 
term in the Actor-Critic loss, we can calculate the entropy of the policy 
distribution and add it to the loss with a regularization coefficient.


"""
### 1.1 Implementation of Entropy regularization term 

To implement entropy regularization, the following changes were made:

- **Modified the get_action method to return the policy distribution (probs) along with the selected action, log probability, and state value**: 
<pre>
``` python
    # in get_action method
    return action_id, log_prob, value, probs
```
</pre>

- **Updated the update_ac method definition to accept an additional argument probs_list**:

<pre>
``` python
    @staticmethod
def update_ac(network, rewards, log_probs, values, masks, Qval, probs_list, gamma=GAMMA):
```
</pre>

- **Calculated the entropy of the policy distributions within the update_ac method:**

<pre>
``` python
    entropy = -torch.mean(torch.sum(probs_list * torch.log(probs_list + 1e-9), dim=-1))
```
</pre>

- **Added an entropy coefficient to balance the importance of the entropy term in the loss calculation**:

<pre>
``` python
    entropy_coefficient = 0.001

```
</pre>

- **Modified the actor-critic loss (ac_loss) calculation to include the entropy term**:

<pre>
``` python
    ac_loss = actor_loss + critic_loss - entropy_coefficient * entropy
```
</pre>


## Asignement 21.02


### 1.0 Modeification of the Rocket env

To implement the current PPO model we had to create a new Rocket environement that you can find in the file `rocket_env_trial.py`and it is basically the rocket environement used by our previous A2C , but in this case inheriting the gym env as well as receiving some modification to the code so that it can be functional. 

### 1.1 Implementation of the algorythm PPO from stable baseline 

You can find the implementation of the PPO in the file `testbed_train_ppo.py`.
We have implmented the PPO algorythm as well as creating a function to get the checkpoints for that PPO algorythm called `@ def save_checkpoint_callback`, thus we can run the model and pursuing the training on weights that have been saved


### 1.2 Chart Explantion

to look at the logs run the following command in your terminal :
!!! BE CARREFULL WITH THE STAR AFTER PPO_* YOU NEED TO REPLACE THAT STAR WITH THE NUMBER RELATED TO YOU TRAINING . LOOK THROUGH YOU DOCUMENT AND CHECK FOR WHICH NUMBER HAVE !!!
<pre>
``` terminal
    tensorboard --logdir ./ppo_tensorboard/PPO_*
</pre>


The first chart shows the episode reward and moving average of the reward as a function of the number of episodes trained. The y-axis represents the reward, and the x-axis represents the number of episodes. The blue line represents the episode reward, while the orange line represents the moving average of the reward over the last 50 episodes. The moving average is a way to smooth out the noise in the episode reward and to better visualize the trend of the reward over time.
![image](https://user-images.githubusercontent.com/55255975/233862829-17b400b6-be63-498a-ae27-d3c556424aa6.png)
<p>
The second chart shows the averaged rewards over a sliding window of size window1 as a function of the training episode number. The y-axis represents the average reward over the sliding window, and the x-axis represents the episode number. The blue line represents the reward smoothed out with a Savitzky-Golay filter of order 3 and window size window0. The orange line represents the averaged rewards over a sliding window of size window1. The shaded area represents the standard deviation of the averaged rewards. The purpose of this chart is to visualize the trend of the averaged reward over a longer time horizon and to see if the algorithm is improving over time.
![image](https://user-images.githubusercontent.com/55255975/233862963-5b54fb1c-0037-4a28-93c7-dbd0442c42a0.png)
<p>


### 1.3 

Explain how to improve average precision missing ....
