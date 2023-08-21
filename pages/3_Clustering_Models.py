##########################################################################################################
###################################### LOADINGS ##########################################################
##########################################################################################################
# Loading Libraries
import streamlit as st

##########################################################################################################
#################################### STREAMLIT PAGE ######################################################
##########################################################################################################
st.set_page_config(page_title='Clustering Models',
                   page_icon='🤖',
                   layout='wide')


# -------------- Sidebar --------------------
st.sidebar.markdown('# Clustering Models Evaluation')


# -------------- Body --------------------

tab1, tab2, = st.tabs(
    ["K-Means", "Affinity Propagation"])

with tab1:
    st.title('K-Means')

with tab2:
    st.title('Affinity Propagation')