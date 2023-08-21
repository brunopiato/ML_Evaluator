##########################################################################################################
###################################### LOADINGS ##########################################################
##########################################################################################################
# Loading Libraries
import streamlit as st

import pandas as pd
import numpy as np

import plotly.express as px

from sklearn import metrics as mt

from sklearn import neighbors as nb
from sklearn import tree as tr
from sklearn import ensemble as en

import utils.classif_functions as classif

# Loading data
# X data
X_train_cla = pd.read_csv('./data/classif/X_training.csv', low_memory=False)
X_test_cla = pd.read_csv('./data/classif/X_test.csv', low_memory=False)
X_val_cla = pd.read_csv('./data/classif/X_validation.csv', low_memory=False)

# y data
y_train_cla = np.ravel(pd.read_csv(
    './data/classif/y_training.csv', low_memory=False))
y_test_cla = np.ravel(pd.read_csv(
    './data/classif/y_test.csv', low_memory=False))
y_val_cla = np.ravel(pd.read_csv(
    './data/classif/y_validation.csv', low_memory=False))


##########################################################################################################
#################################### STREAMLIT PAGE ######################################################
##########################################################################################################
st.set_page_config(page_title='Classification Models',
                   page_icon='🤖',
                   layout='wide')

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["K-Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "Model Comparison"])


# -------------- Sidebar --------------------
st.sidebar.markdown('# Classification Models Evaluation Tool')


# ---------------------------------------------------------------
# --------------------- K-Nearest Neighbors ---------------------
# ---------------------------------------------------------------
with tab1:
    st.header("K-Nearest Neighbors Evaluator")

    expander = st.expander('How does a K-Nearest Neighbors Classifier work?')
    expander.write("""
                This is a K-Nearest Neighbors Classifier algorithm
               """)
    colA, colB = st.columns(2)
    with colA:
        k_neigh = st.slider(label='Define the numbers of neighbors (k) to be used:',
                            min_value=1,
                            max_value=50,
                            value=5)
    with colB:
        st.header('')

    # -------------- Training the model --------------------

    knn_results = classif.knn_training(X_train=X_train_cla,
                                       X_val=X_val_cla,
                                       X_test=X_test_cla,
                                       y_train=y_train_cla,
                                       y_val=y_val_cla,
                                       y_test=y_test_cla,
                                       k_neighbors=k_neigh)

    # -------------- Printing in the page --------------------
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
                              value=knn_results['Train'][0])
                with col7:
                    st.metric(label='Accuracy',
                              value=knn_results['Train'][1])
                with col8:
                    st.metric(label='Recall',
                              value=knn_results['Train'][2])
                with col9:
                    st.metric(label='F1-Score',
                              value=knn_results['Train'][3])

                train_plot = px.bar(x=knn_results.index,
                                    y=knn_results['Train'],
                                    color=knn_results.index,
                                    barmode='overlay',
                                    labels={'x': '', 'y': ''},
                                    template='plotly_dark',
                                    hover_name=knn_results.index)
                train_plot.update_yaxes(range=[0.5, 1.0])
                train_plot.update_layout(showlegend=False)
                st.plotly_chart(train_plot, use_container_width=True)

        with col4:
            with st.container():
                st.title('Validation dataset')
                col10, col11, col12, col13 = st.columns(4)
                with col10:
                    st.metric(label='Precision',
                              value=knn_results['Validation'][0])
                with col11:
                    st.metric(label='Accuracy',
                              value=knn_results['Validation'][1])
                with col12:
                    st.metric(label='Recall',
                              value=knn_results['Validation'][2])
                with col13:
                    st.metric(label='F1-Score',
                              value=knn_results['Validation'][3])

                val_plot = px.bar(x=knn_results.index,
                                  y=knn_results['Validation'],
                                  color=knn_results.index,
                                  barmode='overlay',
                                  labels={'x': '', 'y': ''},
                                  template='plotly_dark',
                                  hover_name=knn_results.index)
                val_plot.update_yaxes(range=[0.5, 1.0])
                val_plot.update_layout(showlegend=False)
                st.plotly_chart(val_plot, use_container_width=True)

        with col5:
            with st.container():
                st.title('Test dataset')
                col14, col15, col16, col17 = st.columns(4)
                with col14:
                    st.metric(label='Precision',
                              value=knn_results['Test'][0])
                with col15:
                    st.metric(label='Accuracy',
                              value=knn_results['Test'][1])
                with col16:
                    st.metric(label='Recall',
                              value=knn_results['Test'][2])
                with col17:
                    st.metric(label='F1-Score',
                              value=knn_results['Test'][3])

                test_plot = px.bar(x=knn_results.index,
                                   y=knn_results['Test'],
                                   color=knn_results.index,
                                   barmode='overlay',
                                   labels={'x': '', 'y': ''},
                                   template='plotly_dark',
                                   hover_name=knn_results.index)
                test_plot.update_yaxes(range=[0.5, 1.0])
                test_plot.update_layout(showlegend=False)
                st.plotly_chart(test_plot, use_container_width=True)


# ---------------------------------------------------------------
# ----------------------- Decision tree -------------------------
# ---------------------------------------------------------------
with tab2:
    st.header("Decision Tree Evaluator")

    expander = st.expander('How does a Decision Tree Classifier work?')
    expander.write("""
                This is a Decision Tree Classifier algorithm
               """)
    colA, colB = st.columns(2)
    with colA:
        m_treedepth = st.slider(label='Define maximum tree depth value:',
                                min_value=1,
                                max_value=200,
                                value=50)
    with colB:
        st.header('')

    # -------------- Training the model --------------------
    dt_results = classif.dt_training(X_train=X_train_cla,
                                     X_val=X_val_cla,
                                     X_test=X_test_cla,
                                     y_train=y_train_cla,
                                     y_val=y_val_cla,
                                     y_test=y_test_cla,
                                     mdepth=m_treedepth)

    # -------------- Printing in the page --------------------
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
                              value=dt_results['Train'][0])
                with col7:
                    st.metric(label='Accuracy',
                              value=dt_results['Train'][1])
                with col8:
                    st.metric(label='Recall',
                              value=dt_results['Train'][2])
                with col9:
                    st.metric(label='F1-Score',
                              value=dt_results['Train'][3])

                train_plot = px.bar(x=dt_results.index,
                                    y=dt_results['Train'],
                                    color=dt_results.index,
                                    barmode='overlay',
                                    labels={'x': '', 'y': ''},
                                    template='plotly_dark',
                                    hover_name=dt_results.index)
                train_plot.update_yaxes(range=[0.5, 1.0])
                train_plot.update_layout(showlegend=False)
                st.plotly_chart(train_plot, use_container_width=True)

        with col4:
            with st.container():
                st.title('Validation dataset')
                col10, col11, col12, col13 = st.columns(4)
                with col10:
                    st.metric(label='Precision',
                              value=dt_results['Validation'][0])
                with col11:
                    st.metric(label='Accuracy',
                              value=dt_results['Validation'][1])
                with col12:
                    st.metric(label='Recall',
                              value=dt_results['Validation'][2])
                with col13:
                    st.metric(label='F1-Score',
                              value=dt_results['Validation'][3])

                val_plot = px.bar(x=dt_results.index,
                                  y=dt_results['Validation'],
                                  color=dt_results.index,
                                  barmode='overlay',
                                  labels={'x': '', 'y': ''},
                                  template='plotly_dark',
                                  hover_name=dt_results.index)
                val_plot.update_yaxes(range=[0.5, 1.0])
                val_plot.update_layout(showlegend=False)
                st.plotly_chart(val_plot, use_container_width=True)

        with col5:
            with st.container():
                st.title('Test dataset')
                col14, col15, col16, col17 = st.columns(4)
                with col14:
                    st.metric(label='Precision',
                              value=dt_results['Test'][0])
                with col15:
                    st.metric(label='Accuracy',
                              value=dt_results['Test'][1])
                with col16:
                    st.metric(label='Recall',
                              value=dt_results['Test'][2])
                with col17:
                    st.metric(label='F1-Score',
                              value=dt_results['Test'][3])

                test_plot = px.bar(x=dt_results.index,
                                   y=dt_results['Test'],
                                   color=dt_results.index,
                                   barmode='overlay',
                                   labels={'x': '', 'y': ''},
                                   template='plotly_dark',
                                   hover_name=dt_results.index)
                test_plot.update_yaxes(range=[0.5, 1.0])
                test_plot.update_layout(showlegend=False)
                st.plotly_chart(test_plot, use_container_width=True)


# ---------------------------------------------------------------
# --------------------- Random Forest ---------------------
# ---------------------------------------------------------------
with tab3:
    st.header("Random Forest Classifier evaluator")

    expander = st.expander('How does a Random Forest Classifier work?')
    expander.write("""
                This is a Random Forest Classifier algorithm
               """)
    colA, colB = st.columns(2)
    with colA:
        m_depth = st.slider(label='Define maximum depth value:',
                            min_value=1,
                            max_value=200,
                            value=50)
    with colB:
        nest = st.slider(label='Define the number of estimators:',
                         min_value=1,
                         max_value=100,
                         value=50)

    # -------------- Training the model --------------------
    rf_results = classif.rf_training(X_train=X_train_cla,
                                     X_val=X_val_cla,
                                     X_test=X_test_cla,
                                     y_train=y_train_cla,
                                     y_val=y_val_cla,
                                     y_test=y_test_cla,
                                     mdepth=m_depth,
                                     n_est=nest)

    # -------------- Printing in the page --------------------
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
                              value=rf_results['Train'][0])
                with col7:
                    st.metric(label='Accuracy',
                              value=rf_results['Train'][1])
                with col8:
                    st.metric(label='Recall',
                              value=rf_results['Train'][2])
                with col9:
                    st.metric(label='F1-Score',
                              value=rf_results['Train'][3])

                train_plot = px.bar(x=rf_results.index,
                                    y=rf_results['Train'],
                                    color=rf_results.index,
                                    barmode='overlay',
                                    labels={'x': '', 'y': ''},
                                    template='plotly_dark',
                                    hover_name=rf_results.index)
                train_plot.update_yaxes(range=[0.5, 1.0])
                train_plot.update_layout(showlegend=False)
                st.plotly_chart(train_plot, use_container_width=True)

        with col4:
            with st.container():
                st.title('Validation dataset')
                col10, col11, col12, col13 = st.columns(4)
                with col10:
                    st.metric(label='Precision',
                              value=rf_results['Validation'][0])
                with col11:
                    st.metric(label='Accuracy',
                              value=rf_results['Validation'][1])
                with col12:
                    st.metric(label='Recall',
                              value=rf_results['Validation'][2])
                with col13:
                    st.metric(label='F1-Score',
                              value=rf_results['Validation'][3])

                val_plot = px.bar(x=rf_results.index,
                                  y=rf_results['Validation'],
                                  color=rf_results.index,
                                  barmode='overlay',
                                  labels={'x': '', 'y': ''},
                                  template='plotly_dark',
                                  hover_name=rf_results.index)
                val_plot.update_yaxes(range=[0.5, 1.0])
                val_plot.update_layout(showlegend=False)
                st.plotly_chart(val_plot, use_container_width=True)

        with col5:
            with st.container():
                st.title('Test dataset')
                col14, col15, col16, col17 = st.columns(4)
                with col14:
                    st.metric(label='Precision',
                              value=rf_results['Test'][0])
                with col15:
                    st.metric(label='Accuracy',
                              value=rf_results['Test'][1])
                with col16:
                    st.metric(label='Recall',
                              value=rf_results['Test'][2])
                with col17:
                    st.metric(label='F1-Score',
                              value=rf_results['Test'][3])

                test_plot = px.bar(x=rf_results.index,
                                   y=rf_results['Test'],
                                   color=rf_results.index,
                                   barmode='overlay',
                                   labels={'x': '', 'y': ''},
                                   template='plotly_dark',
                                   hover_name=rf_results.index)
                test_plot.update_yaxes(range=[0.5, 1.0])
                test_plot.update_layout(showlegend=False)
                st.plotly_chart(test_plot, use_container_width=True)


# ---------------------------------------------------------------
# --------------------- Logistic Regression ---------------------
# ---------------------------------------------------------------
with tab4:
    st.header('Logistic Regression Evaluator')

    expander = st.expander('How does a Logistic Regression Classifier work?')
    expander.write("""
                This is a Logistic Regression Classifier algorithm
               """)

    colA, colB, colC = st.columns(3)
    with colA:
        m_iter = st.slider(label='Define maximum iterations:',
                           min_value=1,
                           max_value=200,
                           value=100)
    with colB:
        nest = st.slider(label='Define the value for C:',
                         step=0.1,
                         min_value=0.1,
                         max_value=1.0,
                         value=1.0)
    with colC:
        s_type = st.selectbox(label='Select the solver type:',
                              options=('newton-cholesky', 'lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'))

    # -------------- Training the model --------------------
    logreg_results = classif.logreg_training(X_train=X_train_cla,
                                             X_val=X_val_cla,
                                             X_test=X_test_cla,
                                             y_train=y_train_cla,
                                             y_val=y_val_cla,
                                             y_test=y_test_cla,
                                             C_value=1.0,
                                             solver_type=s_type,
                                             max_iterations=100)

    # -------------- Printing in the page --------------------
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
                              value=logreg_results['Train'][0])
                with col7:
                    st.metric(label='Accuracy',
                              value=logreg_results['Train'][1])
                with col8:
                    st.metric(label='Recall',
                              value=logreg_results['Train'][2])
                with col9:
                    st.metric(label='F1-Score',
                              value=logreg_results['Train'][3])

                train_plot = px.bar(x=logreg_results.index,
                                    y=logreg_results['Train'],
                                    color=logreg_results.index,
                                    barmode='overlay',
                                    labels={'x': '', 'y': ''},
                                    template='plotly_dark',
                                    hover_name=logreg_results.index)
                train_plot.update_yaxes(range=[0.5, 1.0])
                train_plot.update_layout(showlegend=False)
                st.plotly_chart(train_plot, use_container_width=True)

        with col4:
            with st.container():
                st.title('Validation dataset')
                col10, col11, col12, col13 = st.columns(4)
                with col10:
                    st.metric(label='Precision',
                              value=logreg_results['Validation'][0])
                with col11:
                    st.metric(label='Accuracy',
                              value=logreg_results['Validation'][1])
                with col12:
                    st.metric(label='Recall',
                              value=logreg_results['Validation'][2])
                with col13:
                    st.metric(label='F1-Score',
                              value=logreg_results['Validation'][3])

                val_plot = px.bar(x=logreg_results.index,
                                  y=logreg_results['Validation'],
                                  color=logreg_results.index,
                                  barmode='overlay',
                                  labels={'x': '', 'y': ''},
                                  template='plotly_dark',
                                  hover_name=logreg_results.index)
                val_plot.update_yaxes(range=[0.5, 1.0])
                val_plot.update_layout(showlegend=False)
                st.plotly_chart(val_plot, use_container_width=True)

        with col5:
            with st.container():
                st.title('Test dataset')
                col14, col15, col16, col17 = st.columns(4)
                with col14:
                    st.metric(label='Precision',
                              value=logreg_results['Test'][0])
                with col15:
                    st.metric(label='Accuracy',
                              value=logreg_results['Test'][1])
                with col16:
                    st.metric(label='Recall',
                              value=logreg_results['Test'][2])
                with col17:
                    st.metric(label='F1-Score',
                              value=logreg_results['Test'][3])

                test_plot = px.bar(x=logreg_results.index,
                                   y=logreg_results['Test'],
                                   color=logreg_results.index,
                                   barmode='overlay',
                                   labels={'x': '', 'y': ''},
                                   template='plotly_dark',
                                   hover_name=logreg_results.index)
                test_plot.update_yaxes(range=[0.5, 1.0])
                test_plot.update_layout(showlegend=False)
                st.plotly_chart(test_plot, use_container_width=True)

# ---------------------------------------------------------------
# --------------------- Model Comparison ---------------------
# ---------------------------------------------------------------
with tab5:
    st.header('K-Nearest Neighbors')
    st.table(knn_results.T)
    st.header('Decision Trees')
    st.table(dt_results.T)
    st.header('Random Forest')
    st.table(rf_results.T)
    st.header('Logistic Regression')
    st.table(logreg_results.T)
