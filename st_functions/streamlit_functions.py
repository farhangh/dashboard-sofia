import streamlit as st
import streamlit.components.v1 as components
import requests
import json
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# from lime.lime_tabular import LimeTabularExplainer

# url = "http://127.0.0.1:5000/client_score/?SK_ID_CURR="
#url = "http://127.0.0.1:5000/"
url = "https://api-slask-sofia.herokuapp.com"




def st_title():
    st.title("loan dashboard")


def st_client_id():
    client_id = st.number_input('Client id:',
                                min_value=100002, max_value=152322,
                                value=104405, step=1)
    # URL of the client id  API
    score_url = url + "client_score/?SK_ID_CURR=104405" #+ str(104405) #str(client_id)
    # Requesting the api
    response = requests.get("score_url")
    # Convert from JSON format to Python dict
    content = json.loads(response.content.decode('utf-8'))["score"]["0"]
    if content > -1:
        status = "Loan Accepted" if content == 0 else "Loan Refused"
    else:
        status = "No data available."
    st.markdown(f'<h5 ;'f'font-size:16px;">{"Loan status:&emsp;"}{status}</h1>',
                unsafe_allow_html=True)

    return client_id


def st_buttons(client_id):
    """
    Defines the web page's menus
    :param client_id: int
        Client's id
    :return: button objects
    """
    with st.sidebar:
        b_importance = st.button("Global importance  ")
        b_dist = st.button("Characteristics : id_" + str(client_id))
        b_loc_importance = st.button("Decision explanation : id_" + str(client_id))
        # n_features=st.number_input('Number of features:', min_value=2, max_value=20, value=10, step=1)
        n_features = st.slider('Number of features:', min_value=1, max_value=20, value=10, step=1)
    return b_importance, b_loc_importance, b_dist, n_features


def plot_g_importance(n_features, b_importance):
    """
    Plots the general feature importance.
    :param n_features:
    :param b_importance: button obj
        Characteristics global contribution menu
    :return: feature importance plot
    """
    # n_features = st.number_input('Number of features:', min_value=2, max_value=20, value=10, step=1)
    # URL of the client id  API url = "http://127.0.0.1:5000/"
    g_importance_url = url + "global_importance/?n=" + str(n_features)
    # Requesting the api
    response = requests.get(g_importance_url)
    # Convert from JSON format to Python dict
    content = json.loads(response.content)
    # content = eval(content)
    df = pd.DataFrame(content)

    if b_importance:
        st.write("### Global feature importance")
        sns.set(font_scale=1.5)
        fig = plt.figure(figsize=(15, 25))
        sns.barplot(data=df, x="importance", y="feature")
        st.write(fig)


# if st.checkbox("Seaborn Pairplot",value=True):
#	import seaborn as sns
#	fig = sns.pairplot(df, hue="SKU")
#	st.pyplot(fig)

def plot_l_importance(b_loc_importance, client_id):
    """
    Displays the results of the explain_instance in the Loan allocation menu
    :param client_id:
    :param b_loc_importance: button obj
        Loan allocation menu
    :param client_id: int
        Customer's id
    :return: list
        First (10) important features selected by explain_instance
    """
    l_importance_url = url + 'local_importance/?SK_ID_CURR=' + str(client_id)

    # Requesting the api
    response = requests.get(l_importance_url)
    # Convert from JSON format to Python dict
    content = json.loads(response.content)
    # content = eval(content)
    df = pd.DataFrame(content)

    if b_loc_importance:
        st.write("### Local feature importance")
        sns.set(font_scale=1.5)
        fig = plt.figure(figsize=(15, 25))
        if df.empty:
            st.write('No data')
        else:
            sns.barplot(data=df, x="importance", y="features")
            st.write(fig)

        # display_explanation(explanation, b_loc_importance)


def get_Xtrain(df, features):
    """
    :param df:
    :param features:
    :return:
    """
    y = df["TARGET"].values
    X = df[features].values
    X_sc = StandardScaler().fit_transform(X)
    X_t = train_test_split(X_sc, y, test_size=0.3)[0]
    return X_t, X_sc


def display_explanation(exp, b_loc_importance):
    """
    Display the local feature importance in the webpage
    :param exp: explain_instance obj
        local feature importance computed for a data instance
    :param b_loc_importance: button obj
        Loan allocation menu
    """
    if b_loc_importance:
        st.write("### Local feature importance explanation ")
        exp_html = exp.as_html()
        white_bg = "<style>:root {background-color: white;}</style>"
        text_html = exp_html + white_bg
        st.components.v1.html(html=text_html, height=700)


def select_features(exp):
    """
    param exp: explain_instance obj
        local feature importance computed for a data instance
    :return: list
     First (10) important features selected by explain_instance
    """
    s_features = [feature.split("<")[0] for feature, value in exp.as_list()]
    s_features = [feature.split(">")[0].strip() for feature in s_features]
    return s_features


def dist_per_axis(ax, features, df_target, df_instance):
    """
    Makes a loop over all figure axes and plot the distribution
    :param ax: ndarray
        axe pyplot object
    :param features: list of selected features
    :param df_target: dataframe for either OK class(0) or Risky class(1)
    :param df_instance: dataframe
        Data for the given customer
    """
    for i, feat in enumerate(features):
        axis = ax[int(round(i / 2 + .1)), i % 2]
        hist = axis.hist(df_target[feat], bins=30, log=True)
        axis.set_xlabel(feat, fontsize=18)
        axis.tick_params(axis='both', which='major', labelsize=16)
        axis.tick_params(axis='both', which='minor', labelsize=10)

        # Marking the instance location on the distribution
        axis.plot([df_instance[feat]] * 2, [0, hist[0].max() / 3.], c="r", linewidth=4)


def plot_dist(df, features, target, uid):
    """
    Plot the distribution of the given features for the given target class in the webpage.
    :param df: dataframe
        Whole dataset
    :param features: list of selected features for a given customer (by explain_instance)
    :param target: int
        0 for Accepted, 1 for Refused
    :param uid: int
        Customer's id
    """

    data_target = df[df["TARGET"] == target]
    data_instance = df[df["SK_ID_CURR"] == uid]
    if len(data_instance) == 0:
        st.write("No data available.")
        return 0

    n_row = int(round(len(features) / 2 + .1))
    fig, ax = plt.subplots(n_row, 2, figsize=(15, 20), constrained_layout=True)
    if target == 0:
        fig_title = fig.suptitle("Distributions for an Accepted loan", fontsize=25)
    else:
        fig_title = fig.suptitle("Distributions for a Refused allocation", fontsize=25)

    dist_per_axis(ax, features, data_target, data_instance)
    st.write(fig)


def plot_class_dist(df, features, uid, b_dist):
    """
    Plots the distribution of the given features for the target class separately in the
    "Distribution of characteristics" menu.
    :param df: dataframe Whole dataset
    :param features: list of selected features for a given customer (by explain_instance)
    :param uid: int
        Customer's id
    :param b_dist: button obj
        "Distribution of characteristics" menu
    """
    if b_dist:
        plot_dist(df, features, 0, uid)
        plot_dist(df, features, 1, uid)
