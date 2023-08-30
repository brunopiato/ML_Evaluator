##########################################################################################################
###################################### LOADINGS ##########################################################
##########################################################################################################
# Loading Libraries
import streamlit as st


##########################################################################################################
#################################### STREAMLIT PAGE ######################################################
##########################################################################################################
st.set_page_config(page_title='Home',
                   page_icon='🏠️',
                   layout='centered')


# -------------- Sidebar --------------------
st.sidebar.title('Machine Learning Model Evaluator')
st.sidebar.markdown('#### by Bruno Piato')

st.markdown("""
            # Machine Learning Evaluator
            
            This WebApp was developed to bring to life the final essay of the Machine Learning Fundamentals discipline. It is quite common for data science and analytics teams to build some essays on the functioning of a few different machine learning algorithms in the pursuit to:
            
            - Explore scientifically the algorithms functioning
            - Gain more knowledge and experience about how the algorithms behave
            - Their main hyper-parameters and how they impact the final results
            - Be able to quickly identify the best algorithms to use in each situation
            - Consolidate the studies and learnings through the rational implementation and fine-tuning of several algorithms
            """)

st.markdown("""
            Its main goal is to provide a visual implementation of how the different hyperparameters of some of the main machine learning algorithms impact their performance metrics. All the algorithms of each of the three types were trained, validated and tested using the exact same dataset so the observed differences are only due to algorithms functioning and hyperparameter fine-tuning.
            
            Here I implemented a few of the ML algorithms we used during the discipline:
            - Classification algorithms:
                - K-Nearest Neighbors
                - Decision Trees 
                - Random Forest
                - Logistic Regression
            - Regression algorithms:
                - Linear Model
                - Decision Trees
                - Random Forest
                - Polinomial Regression
                - RANSAC
            - Clustering algorithms:
                - K-Means
                - Affinity Propagation

            At the last tab of each page there is a model comparison of the algorithms as the user set them. 
            
            As all the algorithms are rerun every time the user changes a hyperparameter, it can take a little long to the results be correctly displayed, so please, be patient. 
            
            If you have any question, suggestion, issue report and critics, please let me know by reaching me through [e-mail](mailto:piatobio@gmail.com) or by [Discord](https://discordapp.com/users/438408418429239296) or find me in [LinkedIn](https://www.linkedin.com/in/piatobruno/) or [GitHub](https://github.com/brunopiato/).
            """)
