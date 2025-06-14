import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

st.title("Facebook Ads Optimization with UCB")

uploaded_file = st.file_uploader("Upload your Facebook Ads CSV dataset", type=["csv"])
if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
    
    T = st.number_input("Number of rounds (T)", min_value=10, max_value=1000, value=200, step=10)
    num_ads = dataset.shape[1]
    st.write(f"Number of ads detected: {num_ads}")
    
    if st.button("Run UCB Algorithm"):
        ads_selected = []
        numbers_of_selections = [0] * num_ads
        sums_of_rewards = [0] * num_ads
        total_reward = 0
        cumulative_rewards = []

        for n in range(T):
            ad = 0
            max_upper_bound = 0
            for i in range(num_ads):
                if numbers_of_selections[i] > 0:
                    average_reward = sums_of_rewards[i] / numbers_of_selections[i]
                    delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
                    upper_bound = average_reward + delta_i
                else:
                    upper_bound = 1e400
                if upper_bound > max_upper_bound:
                    max_upper_bound = upper_bound
                    ad = i
            ads_selected.append(ad)
            numbers_of_selections[ad] += 1
            reward = dataset.values[n, ad]
            sums_of_rewards[ad] += reward
            total_reward += reward
            cumulative_rewards.append(total_reward)
        
        st.write(f"### Total reward after {T} rounds: {total_reward}")
        
        # Top 3 ads info
        ad_data = []
        for i in range(num_ads):
            ad_data.append({
                'Ad': i,
                'Number of selections': numbers_of_selections[i],
                'Total reward': sums_of_rewards[i]
            })
        ad_df = pd.DataFrame(ad_data)
        ad_df = ad_df.sort_values(by='Total reward', ascending=False)
        
        st.write("### Top 3 Ads by Total Reward")
        st.dataframe(ad_df.head(3).reset_index(drop=True))
        
        # Histogram plot
        fig1, ax1 = plt.subplots()
        ax1.hist(ads_selected, bins=np.arange(num_ads+1) - 0.5, edgecolor='black')
        ax1.set_title('Histogram of ads selections')
        ax1.set_xlabel('Ads')
        ax1.set_ylabel('Number of times each ad was selected')
        ax1.set_xticks(range(num_ads))
        st.pyplot(fig1)
        
        # Cumulative reward plot
        fig2, ax2 = plt.subplots()
        ax2.plot(range(1, T+1), cumulative_rewards, marker='o')
        ax2.set_title('Cumulative Reward over Rounds')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Cumulative Reward')
        st.pyplot(fig2)

else:
    st.info("Please upload your dataset CSV to get started.")
