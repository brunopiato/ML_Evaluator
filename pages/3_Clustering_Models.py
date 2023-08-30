##########################################################################################################
###################################### LOADINGS ##########################################################
##########################################################################################################
# Loading Libraries
import streamlit as st
import pandas as pd
import plotly.express as px

# Loading dataset
kmeans_result = pd.read_csv(
    './data/clust/kmeans_result.csv', low_memory=False)

affprop_result = pd.read_csv(
    './data/clust/affprop_result.csv', low_memory=False)

##########################################################################################################
#################################### STREAMLIT PAGE ######################################################
##########################################################################################################
st.set_page_config(page_title='Clustering Models',
                   page_icon='🤖',
                   layout='wide')


# -------------- Sidebar --------------------
st.sidebar.markdown('# Clustering Models Evaluation')
st.sidebar.markdown('#### by Bruno Piato')

# -------------- Body --------------------

tab1, tab2, = st.tabs(
    ["K-Means", "Affinity Propagation"])

with tab1:
    st.title('K-Means')
    expander_kmeans = st.expander(
        'How does the K-Means algorithm work?')
    expander_kmeans.write("""
                            The K-Means algorithm is an unsupervised learning algorithm that divides the observations in K different clusters thorugh the estipulation of K different means. It forms K centroids around which are distributed the examples and observations. It iniciatlly calculates K randomly distributed centroid means, defining K clusters according to the distances from the centroid to the example. It will, then, iteratively recalculated the centroids and reassinging the examples to the clusters. Once the clusters stop changing in composition and centroid position, the algorithm is finished and the distance to the centroid will be used to determine to which cluster new observations belong. Despite its methodological simplicity, it demands the user to know a priori the number of clusters in the dataset. To mitigate this caveate, the Elbow Method can be implemented to find the best value of K.
                         """)

    kmeans_plot = px.line(kmeans_result,
                          x='k',
                          y='silhouette_score',
                          labels={'k': 'K Values',
                                  'silhouette_score': 'Silhouette Score'},
                          text='k',
                          template='plotly_dark',
                          width=800, height=400,
                          title='K-Means Model evaluation').update_traces(marker=dict(size=20))

    st.plotly_chart(kmeans_plot, use_container_width=True)

with tab2:
    with st.container():
        st.title('Affinity Propagation')
        expander_affprop = st.expander(
            'How does the Affinity Propagation algorithm work?')
        expander_affprop.write("""
                                The Affinity Propagation algorithm is an unsupervised learning algorithm that clusters the examples of a dataset based on four matrices using Linear Algebra and Graph Theory. The Similarity Matrix between examples is calculated using a distance estimation method. From that matrix two other are iteratively calculated, the Responsability Matrix and the Disponibility Matrix. The first accomodate the chances of two observations belong to the same cluster and the second determines how much an example is available to belong to a different cluster. Finally a Criterion Matrix is computed, where each example row has a probability to belong to the same group as the example on the columns. From this last matrix the clusters are formed. The "preference" hyperparameter determines how much an example is prone to form another fluster. Higher values determine more clusters with fewer participants.
                            """)

        affprop_plot = px.line(affprop_result,
                               x='preference',
                               y='silhouette_score',
                               labels={'preference': 'Preference',
                                       'silhouette_score': 'Silhouette Score'},
                               text='n_clusters',
                               template='plotly_dark',
                               width=1000, height=400,
                               title='Affinity Propagation Model evaluation').update_traces(marker=dict(size=20),
                                                                                            textposition="middle center")
        col1, col2, col3, col4 = st.columns([5, 1, 1, 1])

        with col1:
            affprop_preference = st.slider(label='Define preference value:',
                                           min_value=-100,
                                           max_value=2,
                                           value=-50,
                                           step=2)
            affprop_filtered = affprop_result[affprop_result['preference']
                                              == affprop_preference]

        with col2:
            st.title("")

        with col3:
            affprop_ss = affprop_filtered['silhouette_score']
            st.metric(value=round(affprop_ss, 3), label='Silhouette Score')

        with col4:
            affprop_n_clusters = affprop_filtered['n_clusters']
            st.metric(value=int(affprop_n_clusters),
                      label='Number of clusters')

    with st.container():
        st.plotly_chart(affprop_plot, use_container_width=True)
