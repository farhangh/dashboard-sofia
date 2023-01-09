import streamlit as st
import requests
import json
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from st_functions.streamlit_functions import st_title, st_client_id, st_buttons, plot_g_importance, display_explanation, plot_l_importance

# from lime.lime_tabular import LimeTabularExplainer
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split


st_title()
client_id = st_client_id()
b_importance, b_loc_importance, b_dist, n_features = st_buttons(client_id)
plot_g_importance(n_features, b_importance)
plot_l_importance(b_loc_importance, client_id)






# if __name__ == '__main__':
