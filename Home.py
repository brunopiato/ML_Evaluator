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
            
            This WebApp was developed to bring to life the final essay of the Machine Learning Fundamentals discipline. Its main goal is to provide a visual implementation of how the different hyperparameters of some of the main machine learning algorithms impact their performance metrics. 
            
            Here I implemented a few of the ML algorithms we used during the discipline:
            - Classification algorithms:
                - K-Nearest Neighbors
                - Decision Trees 
                - Random Florest
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

            As the algorithms are rerun everytime the user changes the hyperparameter, it can time a little to the results be shown, so please, be patient. 
            
            If you have any question, suggestion, issue report and critics, please let me know by reaching me through [e-mail](piatobio@gmail.com) or by [Discord](brunopiato).
            """)
