# Thompsons Sampling

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Importing the dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Implementing Thompson Sampling Algorithm from scratch
N = dataset.shape[0]
d = dataset.shape[1]
ads_selected =[]
# Initializing vectors of zeros for numbers of rewards 1 and 0
numbers_of_rewards_1 = [0]*d
numbers_of_rewards_0 = [0]*d
total_rewards = 0
for n in range(0, N):
    selected_ad =0
    max_random = 0
    for ad in range(0, d):
        # implementing Thompson Sampling
        random_beta = random.betavariate(numbers_of_rewards_1[ad]+1, 
                                         numbers_of_rewards_0[ad]+1)
        
        if random_beta> max_random:
            max_random = random_beta
            selected_ad = ad
            
    ads_selected.append(selected_ad)
    reward = dataset.values[n, selected_ad] #Obtain rewards
    if reward == 1:
        numbers_of_rewards_1[selected_ad] +=1
    else:
        numbers_of_rewards_0[selected_ad] +=1

    total_rewards += reward
    # Find the version of ad such that UCB is maximized

# Visualizing results as historgram
plt.hist(ads_selected)
plt.title('Histogram of Ads Selection with Thomson Sampling Algorithm')
plt.xlabel('Ads')
plt.ylabel('# of times each ad was selected')
plt.show()



