# Upper Confidence Bound Algorithm - Ensemble Learning
# Simulation of Ensemble Learning
# To determine which version of the same ads hast better CTR with the optimal amount of exploring

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Importing the dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Implementing UCB from scratch
N = dataset.shape[0]
d = dataset.shape[1]
ads_selected =[]
# Initializing vectors of zeros for numbers of selection and sums of rewards
numbers_of_selections = [0]*d
sums_of_rewards = [0]*d
total_rewards = 0
for n in range(0, N):
    selected_ad =0
    max_upper_bound = -1
    for ad in range(0, d):
        if (numbers_of_selections[ad]>0):
            average_reward = sums_of_rewards[ad]/numbers_of_selections[ad]
            delta_i = math.sqrt(3/2*(math.log(n+1)/numbers_of_selections[ad]))
            upper_bound = average_reward + delta_i
        else: 
            # For the first 10 ads because we have empty lists of numbers of selections and sum of rewards
            upper_bound = 1e400 
            #so that for the other first 10 ads, the for loop will go into the if loop, and update the upper bound
            # while it still keeps the upper bound of other version of ads
        if upper_bound> max_upper_bound:
            max_upper_bound = upper_bound
            selected_ad = ad
    ads_selected.append(selected_ad)
    numbers_of_selections[selected_ad] += 1
    # Append the ads selected for each round n and numbers of selections
    rewards = dataset.values[n, selected_ad]
    sums_of_rewards[selected_ad] += rewards
    total_rewards += rewards
    # Find the version of ad such that UCB is maximized

# Visualizing results as historgram
