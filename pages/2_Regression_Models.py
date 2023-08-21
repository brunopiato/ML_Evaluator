##########################################################################################################
###################################### LOADINGS ##########################################################
##########################################################################################################
# Loading Libraries
import streamlit as st

##########################################################################################################
#################################### STREAMLIT PAGE ######################################################
##########################################################################################################
st.set_page_config(page_title='Regression Models',
                   page_icon='🤖',
                   layout='wide')


# -------------- Sidebar --------------------
st.sidebar.markdown('# Regression Models Evaluation')


# -------------- Body --------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor", "Polinomial Regression", "RANSAC Regressor"])

with tab1:
    st.title('Linear Regression')

with tab2:
    st.title('Decision Tree Regression')

with tab3:
    st.title('Random Forest Regression')

with tab4:
    st.title('Polinomial Regression')

with tab5:
    st.title('RANSAC Regressor')
