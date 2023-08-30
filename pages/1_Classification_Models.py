##########################################################################################################
###################################### LOADINGS ##########################################################
##########################################################################################################
# Loading Libraries
import streamlit as st

import pandas as pd
import numpy as np

import plotly.express as px

# Loading plottings datasets
# KNN
results_train_knn = pd.read_csv(
    './data/classif/results/knn_results_train.csv', low_memory=False, index_col=0)
results_val_knn = pd.read_csv(
    './data/classif/results/knn_results_validation.csv', low_memory=False, index_col=0)
results_test_knn = pd.read_csv(
    './data/classif/results/knn_results_test.csv', low_memory=False, index_col=0)

# Decision tree
results_train_dt = pd.read_csv(
    './data/classif/results/dt_results_train.csv', low_memory=False, index_col=0)
results_val_dt = pd.read_csv(
    './data/classif/results/dt_results_validation.csv', low_memory=False, index_col=0)
results_test_dt = pd.read_csv(
    './data/classif/results/dt_results_test.csv', low_memory=False, index_col=0)

# Random Forest
results_train_rf = pd.read_csv(
    './data/classif/results/rf_results_train.csv', low_memory=False, index_col=0)
results_val_rf = pd.read_csv(
    './data/classif/results/rf_results_validation.csv', low_memory=False, index_col=0)
results_test_rf = pd.read_csv(
    './data/classif/results/rf_results_test.csv', low_memory=False, index_col=0)

# Logistic Regression
results_train_logreg = pd.read_csv(
    './data/classif/results/logreg_results_train.csv', low_memory=False, index_col=0)
results_val_logreg = pd.read_csv(
    './data/classif/results/logreg_results_validation.csv', low_memory=False, index_col=0)
results_test_logreg = pd.read_csv(
    './data/classif/results/logreg_results_test.csv', low_memory=False, index_col=0)

# CONCATANATING DATAFRAMES
# KNN
knn_concat_results = pd.concat(
    [results_train_knn, results_val_knn, results_test_knn], axis=0)

# Decision tree
dt_concat_results = pd.concat(
    [results_train_dt, results_val_dt, results_test_dt], axis=0)

# Random Forest
rf_concat_results = pd.concat(
    [results_train_rf, results_val_rf, results_test_rf], axis=0)

# Logistic Regression
logreg_concat_results = pd.concat(
    [results_train_logreg, results_val_logreg, results_test_logreg], axis=0)


########################################################################
########################## STREAMLIT PAGE ##############################
########################################################################
st.set_page_config(page_title='Classification Models',
                   page_icon='🤖',
                   layout='wide')

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["K-Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "Model Comparison"])


# -------------- Sidebar --------------------
st.sidebar.markdown('# Classification Models Evaluation Tool')
st.sidebar.markdown('#### by Bruno Piato')


# ---------------------------------------------------------------
# --------------------- K-Nearest Neighbors ---------------------
# ---------------------------------------------------------------
with tab1:
    st.header("K-Nearest Neighbors Evaluator")

    expander_knn = st.expander(
        'How does the K-Nearest Neighbors Classifier work?')
    expander_knn.write("""
                The K-Nearest Neighbors Classifier algorithm is a supervised learning algorithm used to predict the classification of a certain observation based on the training features and the distance between examples. It searches for K nearest neighbors in a n-dimensional space to determine the most probable class of that observation, where K is the number of neighbors that will participate in the determination of the prediction and n is the number of features in the dataset. For it uses the distances between the exemples, the dataset must be encoded so the features are in the same magnitude. The Dimensionality Curse might also be a problem in this kind of algorithm along with the training time.
                """)
    colA, colB = st.columns(2)
    with colA:
        st.markdown(
            '*Please note that all the plot y-axes range from 0.5 to 1.0.*')
        k_neigh = st.slider(label='Define the numbers of neighbors (k) to be used:',
                            min_value=1,
                            max_value=15,
                            value=5)

    with colB:
        st.header('')

    # -------------- Preparing the data --------------------
    knn_train_filtered = results_train_knn[results_train_knn['k_value'] == k_neigh]
    knn_val_filtered = results_val_knn[results_val_knn['k_value'] == k_neigh]
    knn_test_filtered = results_test_knn[results_test_knn['k_value'] == k_neigh]
    knn_aux = knn_concat_results[knn_concat_results['k_value'] == k_neigh].loc[:, [
        'precision', 'accuracy', 'recall', 'F1']]
    knn_aux.index = ['train', 'val', 'test']
    knn_aux = knn_aux.T

    # -------------- Ploting the page --------------------
    with st.container():
        col1, col2 = st.columns(2)
        # with col1:
        #     st.metric(label='max_depth',
        #               value=mdepth)
        # with col2:
        #     st.metric(label='n_estimators',
        #               value=n_est)
        col3, col4, col5 = st.columns(3)

        with col3:
            with st.container():
                st.title('Training dataset')

                col6, col7, col8, col9 = st.columns(4)
                with col6:
                    st.metric(label='Precision',
                              value=round(knn_train_filtered['precision'], 3))
                with col7:
                    st.metric(label='Accuracy',
                              value=round(knn_train_filtered['accuracy'], 3))
                with col8:
                    st.metric(label='Recall',
                              value=round(knn_train_filtered['recall'], 3))
                with col9:
                    st.metric(label='F1-Score',
                              value=round(knn_train_filtered['F1'], 3))

                knn_train_plot = px.bar(data_frame=knn_aux,
                                        x=knn_aux.index,
                                        y=knn_aux['train'],
                                        color=knn_aux.index,
                                        template='plotly_dark',
                                        labels={'train': 'Score',
                                                'index': 'Metric'})
                knn_train_plot.update_yaxes(range=[0.5, 1.0])
                knn_train_plot.update_layout(showlegend=False)
                st.plotly_chart(knn_train_plot, use_container_width=True)

        with col4:
            with st.container():
                st.title('Validation dataset')
                col10, col11, col12, col13 = st.columns(4)
                with col10:
                    st.metric(label='Precision',
                              value=round(knn_val_filtered['precision'], 3))
                with col11:
                    st.metric(label='Accuracy',
                              value=round(knn_val_filtered['accuracy'], 3))
                with col12:
                    st.metric(label='Recall',
                              value=round(knn_val_filtered['recall'], 3))
                with col13:
                    st.metric(label='F1-Score',
                              value=round(knn_val_filtered['F1'], 3))

                knn_val_plot = px.bar(data_frame=knn_aux,
                                      x=knn_aux.index,
                                      y=knn_aux['val'],
                                      color=knn_aux.index,
                                      template='plotly_dark',
                                      labels={'val': 'Score',
                                              'index': 'Metric'})
                knn_val_plot.update_yaxes(range=[0.5, 1.0])
                knn_val_plot.update_layout(showlegend=False)
                st.plotly_chart(knn_val_plot, use_container_width=True)

        with col5:
            with st.container():
                st.title('Test dataset')
                col14, col15, col16, col17 = st.columns(4)
                with col14:
                    st.metric(label='Precision',
                              value=round(knn_test_filtered['precision'], 3))
                with col15:
                    st.metric(label='Accuracy',
                              value=round(knn_test_filtered['accuracy'], 3))
                with col16:
                    st.metric(label='Recall',
                              value=round(knn_test_filtered['recall'], 3))
                with col17:
                    st.metric(label='F1-Score',
                              value=round(knn_test_filtered['F1'], 3))

                knn_test_plot = px.bar(data_frame=knn_aux,
                                       x=knn_aux.index,
                                       y=knn_aux['test'],
                                       color=knn_aux.index,
                                       template='plotly_dark',
                                       labels={'test': 'Score',
                                                'index': 'Metric'})
                knn_test_plot.update_yaxes(range=[0.5, 1.0])
                knn_test_plot.update_layout(showlegend=False)
                st.plotly_chart(knn_test_plot, use_container_width=True)

    with st.container():
        col18, col19, col20 = st.columns(3)
        with col18:
            knn_plot_train = px.line(results_train_knn,
                                     x=results_train_knn.index,
                                     y=['precision', 'accuracy', 'recall', 'F1'],
                                     template='plotly_dark',
                                     title="Metrics comparison according to the K-value - Train Dataset",
                                     labels={'value': 'Metric value',
                                             'index': 'K-Value'}).update_layout(legend=dict(title="Metrics"))
            st.plotly_chart(knn_plot_train, use_container_width=True)
        with col19:
            knn_plot_val = px.line(results_val_knn,
                                   x=results_val_knn.index,
                                   y=['precision', 'accuracy', 'recall', 'F1'],
                                   template='plotly_dark',
                                   title="Metrics comparison according to the K-value - Validation Dataset",
                                   labels={'value': 'Metric value',
                                           'index': 'K-Value'}).update_layout(legend=dict(title="Metrics"))
            st.plotly_chart(knn_plot_val, use_container_width=True)
        with col20:
            knn_plot_test = px.line(results_test_knn,
                                    x=results_test_knn.index,
                                    y=['precision', 'accuracy', 'recall', 'F1'],
                                    template='plotly_dark',
                                    title="Metrics comparison according to the K-value - Test Dataset",
                                    labels={'value': 'Metric value',
                                            'index': 'K-Value'}).update_layout(legend=dict(title="Metrics"))
            st.plotly_chart(knn_plot_test, use_container_width=True)


# ---------------------------------------------------------------
# ----------------------- Decision tree -------------------------
# ---------------------------------------------------------------
with tab2:
    st.header("Decision Tree Evaluator")

    expander_dt = st.expander('How does the Decision Tree Classifier work?')
    expander_dt.write("""
                The Decision Tree Classifier algorithm is a supervised learning algorithm used to predict the classification of a certain observation based on the binary subdivion of the examples according to training features values. For every subdivion the algorithm tries to reduce the heterogeneity within the formed sub-groups, which is measerud using Gini's Impurity Index, Entropy functions or Information Gain functions focusing on the target variable. The algorithm ceases when it reaches the maximum tree depth ("max_depth"), the lowest impurity index, the maximum leaves per node and a few other user set parameters. Accordingly to the features-values pair encoutered during the training process, new observations will be accomadated in a certain leaf, givind its classification prediction.
               """)
    colA, colB = st.columns(2)
    with colA:
        st.markdown(
            '*Please note that all the plot y-axes range from 0.5 to 1.0.*')
        m_treedepth = st.slider(label='Define maximum tree depth value:',
                                min_value=5,
                                max_value=100,
                                value=50,
                                step=5)
    with colB:
        st.header('')

    # -------------- Preparing the data --------------------
    dt_train_filtered = results_train_dt[results_train_dt['max_depth'] == m_treedepth]
    dt_val_filtered = results_val_dt[results_val_dt['max_depth']
                                     == m_treedepth]
    dt_test_filtered = results_test_dt[results_test_dt['max_depth'] == m_treedepth]
    dt_aux = dt_concat_results[dt_concat_results['max_depth'] == m_treedepth].loc[:, [
        'precision', 'accuracy', 'recall', 'F1']]
    dt_aux.index = ['train', 'val', 'test']
    dt_aux = dt_aux.T

    # -------------- Ploting the page --------------------

    dt_train_filtered = results_train_dt[results_train_dt['max_depth'] == m_treedepth]
    dt_val_filtered = results_val_dt[results_val_dt['max_depth']
                                     == m_treedepth]
    dt_test_filtered = results_test_dt[results_test_dt['max_depth'] == m_treedepth]

    with st.container():
        col1, col2 = st.columns(2)
        # with col1:
        #     st.metric(label='max_depth',
        #               value=mdepth)
        # with col2:
        #     st.metric(label='n_estimators',
        #               value=n_est)
        col3, col4, col5 = st.columns(3)

        with col3:
            with st.container():
                st.title('Training dataset')

                col6, col7, col8, col9 = st.columns(4)
                with col6:
                    st.metric(label='Precision',
                              value=round(dt_train_filtered['precision'], 3))
                with col7:
                    st.metric(label='Accuracy',
                              value=round(dt_train_filtered['accuracy'], 3))
                with col8:
                    st.metric(label='Recall',
                              value=round(dt_train_filtered['recall'], 3))
                with col9:
                    st.metric(label='F1-Score',
                              value=round(dt_train_filtered['F1'], 3))

                dt_train_plot = px.bar(data_frame=dt_aux,
                                       x=dt_aux.index,
                                       y=dt_aux['train'],
                                       color=dt_aux.index,
                                       template='plotly_dark',
                                       labels={'train': 'Score',
                                                'index': 'Metric'})
                dt_train_plot.update_yaxes(range=[0.5, 1.0])
                dt_train_plot.update_layout(showlegend=False)
                st.plotly_chart(dt_train_plot, use_container_width=True)

        with col4:
            with st.container():
                st.title('Validation dataset')
                col10, col11, col12, col13 = st.columns(4)
                with col10:
                    st.metric(label='Precision',
                              value=round(dt_val_filtered['precision'], 3))
                with col11:
                    st.metric(label='Accuracy',
                              value=round(dt_val_filtered['accuracy'], 3))
                with col12:
                    st.metric(label='Recall',
                              value=round(dt_val_filtered['recall'], 3))
                with col13:
                    st.metric(label='F1-Score',
                              value=round(dt_val_filtered['F1'], 3))

                dt_val_plot = px.bar(data_frame=dt_aux,
                                     x=dt_aux.index,
                                     y=dt_aux['val'],
                                     color=dt_aux.index,
                                     template='plotly_dark',
                                     labels={'val': 'Score',
                                             'index': 'Metric'})
                dt_val_plot.update_yaxes(range=[0.5, 1.0])
                dt_val_plot.update_layout(showlegend=False)
                st.plotly_chart(dt_val_plot, use_container_width=True)

        with col5:
            with st.container():
                st.title('Test dataset')
                col14, col15, col16, col17 = st.columns(4)
                with col14:
                    st.metric(label='Precision',
                              value=round(dt_test_filtered['precision'], 3))
                with col15:
                    st.metric(label='Accuracy',
                              value=round(dt_test_filtered['accuracy'], 3))
                with col16:
                    st.metric(label='Recall',
                              value=round(dt_test_filtered['recall'], 3))
                with col17:
                    st.metric(label='F1-Score',
                              value=round(dt_test_filtered['F1'], 3))

                dt_test_plot = px.bar(data_frame=dt_aux,
                                      x=dt_aux.index,
                                      y=dt_aux['test'],
                                      color=dt_aux.index,
                                      template='plotly_dark',
                                      labels={'test': 'Score',
                                              'index': 'Metric'})
                dt_test_plot.update_yaxes(range=[0.5, 1.0])
                dt_test_plot.update_layout(showlegend=False)
                st.plotly_chart(dt_test_plot, use_container_width=True)

    with st.container():
        col18, col19, col20 = st.columns(3)
        with col18:
            dt_plot_train = px.line(results_train_dt,
                                    x=results_train_dt.index,
                                    y=['precision', 'accuracy', 'recall', 'F1'],
                                    template='plotly_dark',
                                    title="Metrics comparison according to the max_depth - Train Dataset",
                                    labels={'value': 'Metric value',
                                             'index': 'max_depth'}).update_layout(legend=dict(title="Metrics"))
            st.plotly_chart(dt_plot_train, use_container_width=True)
        with col19:
            dt_plot_val = px.line(results_val_dt,
                                  x=results_val_dt.index,
                                  y=['precision', 'accuracy', 'recall', 'F1'],
                                  template='plotly_dark',
                                  title="Metrics comparison according to the max_depth - Validation Dataset",
                                  labels={'value': 'Metric value',
                                           'index': 'max_depth'}).update_layout(legend=dict(title="Metrics"))
            st.plotly_chart(dt_plot_val, use_container_width=True)
        with col20:
            st.title('')
            dt_plot_test = px.line(results_test_dt,
                                   x=results_test_dt.index,
                                   y=['precision', 'accuracy', 'recall', 'F1'],
                                   template='plotly_dark',
                                   title="Metrics comparison according to the max_depth - Test Dataset",
                                   labels={'value': 'Metric value',
                                            'index': 'max_depth'}).update_layout(legend=dict(title="Metrics"))
            st.plotly_chart(dt_plot_test, use_container_width=True)


# ---------------------------------------------------------------
# --------------------- Random Forest ---------------------------
# ---------------------------------------------------------------
with tab3:
    st.header("Random Forest Classifier evaluator")

    expander_rf = st.expander('How does the Random Forest Classifier work?')
    expander_rf.write("""
                The Random Forest Classifier algorithm is a supervised learning algorithm used to predict the classification of a certain observation based on the consensus obtained by the training of several different Decision Tree algorithms, amplifying the confidence on the predictions. It implements a bootstrap method to generate pseudoreplicates of the dataset to train the trees increasing the variability of the forest, which could be used to calculate the relative importance of each feature. After training n trees from n pseudoreplicates ("n_estimators"), every new observation will be accomodated in a single leaf of each tree and then the mode will be calculated to elect the class prediction.
               """)
    colA, colB = st.columns(2)
    with colA:
        st.markdown(
            '*Please note that all the plot y-axes range from 0.5 to 1.0.*')
        m_depth = st.slider(label='Define maximum depth value:',
                            min_value=5,
                            max_value=100,
                            value=50,
                            step=5)
    with colB:
        st.title('')
        # nest = st.slider(label='Define the number of estimators:',
        #                  min_value=1,
        #                  max_value=100,
        #                  value=50)

    # -------------- Preparing the data --------------------
    rf_train_filtered = results_train_rf[results_train_rf['max_depth'] == m_depth]
    rf_val_filtered = results_val_rf[results_val_rf['max_depth'] == m_depth]
    rf_test_filtered = results_test_rf[results_test_rf['max_depth'] == m_depth]
    rf_aux = rf_concat_results[rf_concat_results['max_depth'] == m_depth].loc[:, [
        'precision', 'accuracy', 'recall', 'F1']]
    rf_aux.index = ['train', 'val', 'test']
    rf_aux = rf_aux.T

    # -------------- Ploting the page --------------------
    # [TODO] Add a filter for the number of estimator trees. To do so I need to rerun the model with a nested for loop
    rf_train_filtered = results_train_rf[results_train_rf['max_depth'] == m_depth]
    rf_val_filtered = results_val_rf[results_val_rf['max_depth']
                                     == m_depth]
    rf_test_filtered = results_test_rf[results_test_rf['max_depth'] == m_depth]

    with st.container():
        col1, col2 = st.columns(2)
        # with col1:
        #     st.metric(label='max_depth',
        #               value=mdepth)
        # with col2:
        #     st.metric(label='n_estimators',
        #               value=n_est)
        col3, col4, col5 = st.columns(3)

        with col3:
            with st.container():
                st.title('Training dataset')

                col6, col7, col8, col9 = st.columns(4)
                with col6:
                    st.metric(label='Precision',
                              value=round(rf_train_filtered['precision'], 3))
                with col7:
                    st.metric(label='Accuracy',
                              value=round(rf_train_filtered['accuracy'], 3))
                with col8:
                    st.metric(label='Recall',
                              value=round(rf_train_filtered['recall'], 3))
                with col9:
                    st.metric(label='F1-Score',
                              value=round(rf_train_filtered['F1'], 3))

                rf_train_plot = px.bar(data_frame=rf_aux,
                                       x=rf_aux.index,
                                       y=rf_aux['train'],
                                       color=rf_aux.index,
                                       template='plotly_dark',
                                       labels={'train': 'Score',
                                                'index': 'Metric'})
                rf_train_plot.update_yaxes(range=[0.5, 1.0])
                rf_train_plot.update_layout(showlegend=False)
                st.plotly_chart(rf_train_plot, use_container_width=True)

        with col4:
            with st.container():
                st.title('Validation dataset')
                col10, col11, col12, col13 = st.columns(4)
                with col10:
                    st.metric(label='Precision',
                              value=round(rf_val_filtered['precision'], 3))
                with col11:
                    st.metric(label='Accuracy',
                              value=round(rf_val_filtered['accuracy'], 3))
                with col12:
                    st.metric(label='Recall',
                              value=round(rf_val_filtered['recall'], 3))
                with col13:
                    st.metric(label='F1-Score',
                              value=round(rf_val_filtered['F1'], 3))

                rf_val_plot = px.bar(data_frame=rf_aux,
                                     x=rf_aux.index,
                                     y=rf_aux['val'],
                                     color=rf_aux.index,
                                     template='plotly_dark',
                                     labels={'val': 'Score',
                                             'index': 'Metric'})
                rf_val_plot.update_yaxes(range=[0.5, 1.0])
                rf_val_plot.update_layout(showlegend=False)
                st.plotly_chart(rf_val_plot, use_container_width=True)

        with col5:
            with st.container():
                st.title('Test dataset')
                col14, col15, col16, col17 = st.columns(4)
                with col14:
                    st.metric(label='Precision',
                              value=round(rf_test_filtered['precision'], 3))
                with col15:
                    st.metric(label='Accuracy',
                              value=round(rf_test_filtered['accuracy'], 3))
                with col16:
                    st.metric(label='Recall',
                              value=round(rf_test_filtered['recall'], 3))
                with col17:
                    st.metric(label='F1-Score',
                              value=round(rf_test_filtered['F1'], 3))

                rf_test_plot = px.bar(data_frame=rf_aux,
                                      x=rf_aux.index,
                                      y=rf_aux['test'],
                                      color=rf_aux.index,
                                      template='plotly_dark',
                                      labels={'test': 'Score',
                                              'index': 'Metric'})
                rf_test_plot.update_yaxes(range=[0.5, 1.0])
                rf_test_plot.update_layout(showlegend=False)
                st.plotly_chart(rf_test_plot, use_container_width=True)

    with st.container():
        col18, col19, col20 = st.columns(3)
        with col18:
            st.title('')
            rf_plot_train = px.line(results_train_rf,
                                    x=results_train_rf.index,
                                    y=['precision', 'accuracy', 'recall', 'F1'],
                                    template='plotly_dark',
                                    title="Metrics comparison according to the max_depth - Train Dataset",
                                    labels={'value': 'Metric value',
                                             'index': 'max_depth'}).update_layout(legend=dict(title="Metrics"))
            st.plotly_chart(rf_plot_train, use_container_width=True)

        with col19:
            st.title('')
            rf_plot_val = px.line(results_val_rf,
                                  x=results_val_rf.index,
                                  y=['precision', 'accuracy', 'recall', 'F1'],
                                  template='plotly_dark',
                                  title="Metrics comparison according to the max_depth - Validation Dataset",
                                  labels={'value': 'Metric value',
                                           'index': 'max_depth'}).update_layout(legend=dict(title="Metrics"))
            st.plotly_chart(rf_plot_val, use_container_width=True)

        with col20:
            st.title('')
            rf_plot_test = px.line(results_test_rf,
                                   x=results_test_rf.index,
                                   y=['precision', 'accuracy', 'recall', 'F1'],
                                   template='plotly_dark',
                                   title="Metrics comparison according to the max_depth - Test Dataset",
                                   labels={'value': 'Metric value',
                                            'index': 'max_depth'}).update_layout(legend=dict(title="Metrics"))
            st.plotly_chart(rf_plot_test, use_container_width=True)

# ---------------------------------------------------------------
# --------------------- Logistic Regression ---------------------
# ---------------------------------------------------------------
with tab4:
    st.header('Logistic Regression Evaluator')

    expander_logreg = st.expander(
        'How does the Logistic Regression Classifier work?')
    expander_logreg.write("""
                The Logistic Regression Classifier algorithm is a supervised learning algorithm used to predict the classification of a certain observation based on the logistic function (sigmoid function) to predict its class. It uses the maximum likelihood function to optimize the parameters of the logistic function, which will attribute a value between 0 and 1 to the probability of a certain observation belong to a specific class. A cut threshold can be determined (i.e. 0.5) above which the observation will be classified as class A and bellow which it will be classified as class B, for example. The hyperparameter "C" stands for a regularization parameter, where high values tend to weight more the variables.
               """)

    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown(
            '*Please note that all the plot y-axes range from 0.5 to 1.0.*')
        C_val = st.slider(label='Define the value for C:',
                          step=0.1,
                          min_value=0.1,
                          max_value=1.0,
                          value=1.0)
    with colB:
        st.title('')
        # m_iter = st.slider(label='Define maximum iterations:',
        #                    min_value=1,
        #                    max_value=200,
        #                    value=100)
    with colC:
        st.title('')
        # s_type = st.selectbox(label='Select the solver type:',
        #                       options=('newton-cholesky', 'lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'))

    # -------------- Preparing the data --------------------
    logreg_train_filtered = results_train_logreg[results_train_logreg['C_value'] == C_val]
    logreg_val_filtered = results_val_logreg[results_val_logreg['C_value'] == C_val]
    logreg_test_filtered = results_test_logreg[results_test_logreg['C_value'] == C_val]
    logreg_aux = logreg_concat_results[logreg_concat_results['C_value'] == C_val].loc[:, [
        'precision', 'accuracy', 'recall', 'F1']]
    logreg_aux.index = ['train', 'val', 'test']
    logreg_aux = logreg_aux.T

    # -------------- Ploting the page --------------------

    logreg_train_filtered = results_train_logreg[results_train_logreg['C_value'] == C_val]
    logreg_val_filtered = results_val_logreg[results_val_logreg['C_value']
                                             == C_val]
    logreg_test_filtered = results_test_logreg[results_test_logreg['C_value'] == C_val]

    with st.container():
        col1, col2 = st.columns(2)
        # with col1:
        #     st.metric(label='max_depth',
        #               value=mdepth)
        # with col2:
        #     st.metric(label='n_estimators',
        #               value=n_est)
        col3, col4, col5 = st.columns(3)

        with col3:
            with st.container():
                st.title('Training dataset')

                col6, col7, col8, col9 = st.columns(4)
                with col6:
                    st.metric(label='Precision',
                              value=round(logreg_train_filtered['precision'], 3))
                with col7:
                    st.metric(label='Accuracy',
                              value=round(logreg_train_filtered['accuracy'], 3))
                with col8:
                    st.metric(label='Recall',
                              value=round(logreg_train_filtered['recall'], 3))
                with col9:
                    st.metric(label='F1-Score',
                              value=round(logreg_train_filtered['F1'], 3))

                logreg_train_plot = px.bar(data_frame=logreg_aux,
                                           x=logreg_aux.index,
                                           y=logreg_aux['train'],
                                           color=logreg_aux.index,
                                           template='plotly_dark',
                                           labels={'train': 'Score',
                                                   'index': 'Metric'})
                logreg_train_plot.update_yaxes(range=[0.5, 1.0])
                logreg_train_plot.update_layout(showlegend=False)
                st.plotly_chart(logreg_train_plot, use_container_width=True)

        with col4:
            with st.container():
                st.title('Validation dataset')
                col10, col11, col12, col13 = st.columns(4)
                with col10:
                    st.metric(label='Precision',
                              value=round(logreg_val_filtered['precision'], 3))
                with col11:
                    st.metric(label='Accuracy',
                              value=round(logreg_val_filtered['accuracy'], 3))
                with col12:
                    st.metric(label='Recall',
                              value=round(logreg_val_filtered['recall'], 3))
                with col13:
                    st.metric(label='F1-Score',
                              value=round(logreg_val_filtered['F1'], 3))

                logreg_val_plot = px.bar(data_frame=logreg_aux,
                                         x=logreg_aux.index,
                                         y=logreg_aux['val'],
                                         color=logreg_aux.index,
                                         template='plotly_dark',
                                         labels={'val': 'Score',
                                                 'index': 'Metric'})
                logreg_val_plot.update_yaxes(range=[0.5, 1.0])
                logreg_val_plot.update_layout(showlegend=False)
                st.plotly_chart(logreg_val_plot, use_container_width=True)

        with col5:
            with st.container():
                st.title('Test dataset')
                col14, col15, col16, col17 = st.columns(4)
                with col14:
                    st.metric(label='Precision',
                              value=round(logreg_test_filtered['precision'], 3))
                with col15:
                    st.metric(label='Accuracy',
                              value=round(logreg_test_filtered['accuracy'], 3))
                with col16:
                    st.metric(label='Recall',
                              value=round(logreg_test_filtered['recall'], 3))
                with col17:
                    st.metric(label='F1-Score',
                              value=round(logreg_test_filtered['F1'], 3))

                logreg_test_plot = px.bar(data_frame=logreg_aux,
                                          x=logreg_aux.index,
                                          y=logreg_aux['test'],
                                          color=logreg_aux.index,
                                          template='plotly_dark',
                                          labels={'test': 'Score',
                                                  'index': 'Metric'})
                logreg_test_plot.update_yaxes(range=[0.5, 1.0])
                logreg_test_plot.update_layout(showlegend=False)
                st.plotly_chart(logreg_test_plot, use_container_width=True)

    with st.container():
        col18, col19, col20 = st.columns(3)
        with col18:
            logreg_plot_train = px.line(results_train_logreg,
                                        x=results_train_logreg.index,
                                        y=['precision', 'accuracy',
                                            'recall', 'F1'],
                                        template='plotly_dark',
                                        title="Metrics comparison according to the C value - Train Dataset",
                                        labels={'value': 'Metric value',
                                                'index': 'C value'}).update_layout(legend=dict(title="Metrics"))
            st.plotly_chart(logreg_plot_train, use_container_width=True)
        with col19:
            logreg_plot_val = px.line(results_val_logreg,
                                      x=results_val_logreg.index,
                                      y=['precision', 'accuracy', 'recall', 'F1'],
                                      template='plotly_dark',
                                      title="Metrics comparison according to the C value - Validation Dataset",
                                      labels={'value': 'Metric value',
                                              'index': 'C value'}).update_layout(legend=dict(title="Metrics"))
            st.plotly_chart(logreg_plot_val, use_container_width=True)
        with col20:
            logreg_plot_test = px.line(results_test_logreg,
                                       x=results_test_logreg.index,
                                       y=['precision', 'accuracy',
                                           'recall', 'F1'],
                                       template='plotly_dark',
                                       title="Metrics comparison according to the C value - Test Dataset",
                                       labels={'value': 'Metric value',
                                               'index': 'C value'}).update_layout(legend=dict(title="Metrics"))
            st.plotly_chart(logreg_plot_test, use_container_width=True)


# ---------------------------------------------------------------
# --------------------- Model Comparison ---------------------
# ---------------------------------------------------------------
with tab5:
    expander_metrics = st.expander("Whats do these metrics mean?")
    expander_metrics.write(r"""
                           **Precision**: is the percentage of correct predictions of a certain classe an algorithm made given the number of times it predicted that class. Based on the Confusion Matrix it is the number of True Positives (TP) in relation to the number of Positives (both True and False, TP and FP). 
                           
                           $Precision = TP/TP+FP$
                           
                           ---
                           
                           **Accuracy**: is the percentage of correct predictions an algorithm made given all predictions it made. Based on the Confusion Matrix it is the True Negative (TN) plus the True Positive(TP) in relation to the number os predictions the algorithm made. Its simply the percentage of right predictions. It may be misleading when classes are unbalanced.
                           
                           $Accuracy = TP+TF/Predictions$
                           
                           ---
                           
                           **Recall**: is the percentage of a certain class that the algorithm successfully identified as such. Based on the Confusion Matrix it is the number of True Positives (TP) in relation to the number of True Positives (right identifications, TP) plus the number of False Negatives (wrong identifications, FN).
                           
                           $Recall = TP/TP+FN$
                           
                           ---
                           
                           **F1-Score**: is a way the gather both Precision and Recall in a single metric given the Precision-Recall trade-off. It is the harmonic mean between these two metrics weighting relatively more the metric with lowest value.
                           
                           $F1Score = \frac{2}{\frac{1}{Precision} + \frac{1}{Recall}}$
                           """)

    st.markdown('## K-Nearest Neighbors')
    st.table(knn_concat_results[knn_concat_results['k_value'] == k_neigh])

    st.markdown('## Decision Trees')
    st.table(dt_concat_results[dt_concat_results['max_depth'] == m_treedepth])

    st.markdown('## Random Forest')
    st.table(rf_concat_results[rf_concat_results['max_depth'] == m_depth])

    st.markdown('## Logistic Regression')
    st.table(logreg_concat_results[logreg_concat_results['C_value'] == C_val])
