def knn_training(X_train, X_val, X_test, y_train, y_val, y_test, k_neighbors):
    # Import Libraries
    import sklearn.metrics as mt
    import sklearn.neighbors as nb
    import numpy as np
    import pandas as pd

    # Call and Fit
    knn = nb.KNeighborsClassifier(n_neighbors=k_neighbors)
    knn.fit(X_train, y_train)

    # Predicting
    ypred_train_knn = knn.predict(X_train)
    ypred_val_knn = knn.predict(X_val)
    ypred_test_knn = knn.predict(X_test)

    # Getting train metrics
    knn_train_precision = mt.precision_score(y_train, ypred_train_knn)
    knn_train_accuracy = mt.accuracy_score(y_train, ypred_train_knn)
    knn_train_recall = mt.recall_score(y_train, ypred_train_knn)
    knn_train_F1 = mt.f1_score(y_train, ypred_train_knn)

    # Getting validation metrics
    knn_val_precision = mt.precision_score(y_val, ypred_val_knn)
    knn_val_accuracy = mt.accuracy_score(y_val, ypred_val_knn)
    knn_val_recall = mt.recall_score(y_val, ypred_val_knn)
    knn_val_F1 = mt.f1_score(y_val, ypred_val_knn)

    # Getting test metrics
    knn_test_precision = mt.precision_score(y_test, ypred_test_knn)
    knn_test_accuracy = mt.accuracy_score(y_test, ypred_test_knn)
    knn_test_recall = mt.recall_score(y_test, ypred_test_knn)
    knn_test_F1 = mt.f1_score(y_test, ypred_test_knn)

    # Transform results in arrays and DataFrames
    knn_train_results = np.array(
        [round(knn_train_precision, 3), round(knn_train_accuracy, 3), round(knn_train_recall, 3), round(knn_train_F1, 3)])
    knn_val_results = np.array(
        [round(knn_val_precision, 3), round(knn_val_accuracy, 3), round(knn_val_recall, 3), round(knn_val_F1, 3)])
    knn_test_results = np.array(
        [round(knn_test_precision, 3), round(knn_test_accuracy, 3), round(knn_test_recall, 3), round(knn_test_F1, 3)])

    results_knn = pd.DataFrame([knn_train_results, knn_val_results, knn_test_results], columns=[
        'Precision', 'Accuracy', 'Recall', 'F1-Score'], index=['Train', 'Validation', 'Test']).T

    return results_knn


def dt_training(X_train, X_val, X_test, y_train, y_val, y_test, mdepth):
    # Import Libraries
    import sklearn.tree as tree
    import sklearn.metrics as mt
    import pandas as pd
    import numpy as np

    # Call and Fit
    dt_class = tree.DecisionTreeClassifier(max_depth=mdepth,
                                           random_state=42)
    dt_class.fit(X_train, y_train)

    # Predicting
    ypred_train_rf = dt_class.predict(X_train)
    ypred_val_rf = dt_class.predict(X_val)
    ypred_test_rf = dt_class.predict(X_test)

    # Getting train metrics
    dt_class_train_precision = mt.precision_score(y_train, ypred_train_rf)
    dt_class_train_accuracy = mt.accuracy_score(y_train, ypred_train_rf)
    dt_class_train_recall = mt.recall_score(y_train, ypred_train_rf)
    dt_class_train_F1 = mt.f1_score(y_train, ypred_train_rf)
    # Getting validation metrics
    dt_class_val_precision = mt.precision_score(y_val, ypred_val_rf)
    dt_class_val_accuracy = mt.accuracy_score(y_val, ypred_val_rf)
    dt_class_val_recall = mt.recall_score(y_val, ypred_val_rf)
    dt_class_val_F1 = mt.f1_score(y_val, ypred_val_rf)
    # Getting test metrics
    dt_class_test_precision = mt.precision_score(y_test, ypred_test_rf)
    dt_class_test_accuracy = mt.accuracy_score(y_test, ypred_test_rf)
    dt_class_test_recall = mt.recall_score(y_test, ypred_test_rf)
    dt_class_test_F1 = mt.f1_score(y_test, ypred_test_rf)

    # Transform results in arrays and DataFrames
    dt_class_train_results = np.array(
        [round(dt_class_train_precision, 3), round(dt_class_train_accuracy, 3), round(dt_class_train_recall, 3), round(dt_class_train_F1, 3)])
    dt_class_val_results = np.array(
        [round(dt_class_val_precision, 3), round(dt_class_val_accuracy, 3), round(dt_class_val_recall, 3), round(dt_class_val_F1, 3)])
    dt_class_test_results = np.array(
        [round(dt_class_test_precision, 3), round(dt_class_test_accuracy, 3), round(dt_class_test_recall, 3), round(dt_class_test_F1, 3)])

    results_dt_class = pd.DataFrame([dt_class_train_results, dt_class_val_results, dt_class_test_results], columns=[
        'Precision', 'Accuracy', 'Recall', 'F1-Score'], index=['Train', 'Validation', 'Test']).T

    return results_dt_class


def rf_training(X_train, X_val, X_test, y_train, y_val, y_test, mdepth, n_est):
    # Import Libraries
    import sklearn.ensemble as en
    import sklearn.metrics as mt
    import pandas as pd
    import numpy as np

    # Call and Fit
    rf_class = en.RandomForestClassifier(max_depth=mdepth,
                                         n_estimators=n_est,
                                         random_state=42)
    rf_class.fit(X_train, y_train)

    # Predicting
    ypred_train_rf = rf_class.predict(X_train)
    ypred_val_rf = rf_class.predict(X_val)
    ypred_test_rf = rf_class.predict(X_test)

    # Getting train metrics
    rf_class_train_precision = mt.precision_score(y_train, ypred_train_rf)
    rf_class_train_accuracy = mt.accuracy_score(y_train, ypred_train_rf)
    rf_class_train_recall = mt.recall_score(y_train, ypred_train_rf)
    rf_class_train_F1 = mt.f1_score(y_train, ypred_train_rf)
    # Getting validation metrics
    rf_class_val_precision = mt.precision_score(y_val, ypred_val_rf)
    rf_class_val_accuracy = mt.accuracy_score(y_val, ypred_val_rf)
    rf_class_val_recall = mt.recall_score(y_val, ypred_val_rf)
    rf_class_val_F1 = mt.f1_score(y_val, ypred_val_rf)
    # Getting test metrics
    rf_class_test_precision = mt.precision_score(y_test, ypred_test_rf)
    rf_class_test_accuracy = mt.accuracy_score(y_test, ypred_test_rf)
    rf_class_test_recall = mt.recall_score(y_test, ypred_test_rf)
    rf_class_test_F1 = mt.f1_score(y_test, ypred_test_rf)

    # Transform results in arrays and DataFrames
    rf_class_train_results = np.array(
        [round(rf_class_train_precision, 3), round(rf_class_train_accuracy, 3), round(rf_class_train_recall, 3), round(rf_class_train_F1, 3)])
    rf_class_val_results = np.array(
        [round(rf_class_val_precision, 3), round(rf_class_val_accuracy, 3), round(rf_class_val_recall, 3), round(rf_class_val_F1, 3)])
    rf_class_test_results = np.array(
        [round(rf_class_test_precision, 3), round(rf_class_test_accuracy, 3), round(rf_class_test_recall, 3), round(rf_class_test_F1, 3)])

    results_rf_class = pd.DataFrame([rf_class_train_results, rf_class_val_results, rf_class_test_results], columns=[
        'Precision', 'Accuracy', 'Recall', 'F1-Score'], index=['Train', 'Validation', 'Test']).T

    return results_rf_class


def logreg_training(X_train, X_val, X_test, y_train, y_val, y_test, C_value, solver_type, max_iterations):
    # Import Libraries
    import sklearn.linear_model as lm
    import sklearn.metrics as mt
    import pandas as pd
    import numpy as np

    # Call and Fit
    logreg_class = lm.LogisticRegression(C=C_value,
                                         solver=solver_type,
                                         max_iter=max_iterations)
    logreg_class.fit(X_train, y_train)

    # Predicting
    ypred_train_rf = logreg_class.predict(X_train)
    ypred_val_rf = logreg_class.predict(X_val)
    ypred_test_rf = logreg_class.predict(X_test)

    # Getting train metrics
    logreg_class_train_precision = mt.precision_score(y_train, ypred_train_rf)
    logreg_class_train_accuracy = mt.accuracy_score(y_train, ypred_train_rf)
    logreg_class_train_recall = mt.recall_score(y_train, ypred_train_rf)
    logreg_class_train_F1 = mt.f1_score(y_train, ypred_train_rf)
    # Getting validation metrics
    logreg_class_val_precision = mt.precision_score(y_val, ypred_val_rf)
    logreg_class_val_accuracy = mt.accuracy_score(y_val, ypred_val_rf)
    logreg_class_val_recall = mt.recall_score(y_val, ypred_val_rf)
    logreg_class_val_F1 = mt.f1_score(y_val, ypred_val_rf)
    # Getting test metrics
    logreg_class_test_precision = mt.precision_score(y_test, ypred_test_rf)
    logreg_class_test_accuracy = mt.accuracy_score(y_test, ypred_test_rf)
    logreg_class_test_recall = mt.recall_score(y_test, ypred_test_rf)
    logreg_class_test_F1 = mt.f1_score(y_test, ypred_test_rf)

    # Transform results in arrays and DataFrames
    logreg_class_train_results = np.array(
        [round(logreg_class_train_precision, 3), round(logreg_class_train_accuracy, 3), round(logreg_class_train_recall, 3), round(logreg_class_train_F1, 3)])
    logreg_class_val_results = np.array(
        [round(logreg_class_val_precision, 3), round(logreg_class_val_accuracy, 3), round(logreg_class_val_recall, 3), round(logreg_class_val_F1, 3)])
    logreg_class_test_results = np.array(
        [round(logreg_class_test_precision, 3), round(logreg_class_test_accuracy, 3), round(logreg_class_test_recall, 3), round(logreg_class_test_F1, 3)])

    results_logreg_class = pd.DataFrame([logreg_class_train_results, logreg_class_val_results, logreg_class_test_results], columns=[
        'Precision', 'Accuracy', 'Recall', 'F1-Score'], index=['Train', 'Validation', 'Test']).T

    return results_logreg_class
