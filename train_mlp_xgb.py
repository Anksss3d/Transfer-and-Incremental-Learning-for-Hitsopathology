import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import CustomDataset
import numpy as np
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from utilities import calculate_metrics2, generic_dir

def fit_mlp_classifier(x_train, y_train, x_test, y_test, simple=True, NUM_EPOCHS=50, plot_path = None):
    """
    :param x_train: Training data features
    :param y_train: Training data labels
    :param x_test: Testing data features
    :param y_test: Testing data labels
    :param simple: Boolean value specifying whether to use simple MLP model (default=True)
    :param NUM_EPOCHS: Number of epochs for training the MLP model (default=50)
    :param plot_path: Path to save the plot of metrics (optional)
    :return: Tuple containing training accuracy and validation accuracy

    This method fits a multilayer perceptron (MLP) classifier on the given training data and labels. It uses the MLPClassifier class from the sklearn.neural_network module. The MLPClassifier
    * is trained using stochastic gradient descent (sgd) as the solver, with a random state of 42 and an initial learning rate of 0.01. The MLP has two hidden layers with sizes (1024, 512
    *).

    If the 'simple' parameter is True, the MLP is trained using the entire training data and evaluated on the testing data. It calculates the training accuracy, validation accuracy, and
    * predicts the classes for the testing data. It also computes the predicted probabilities for each class.

    If the 'simple' parameter is False, the MLP is trained iteratively using partial_fit function. Each epoch, the MLP is trained on a single batch of data and updated weights are stored
    * for the next epoch. It calculates the training accuracy, validation accuracy, training loss, and validation loss for each epoch. The metrics are printed for each epoch.

    If 'plot_path' is specified, the method calls calculate_metrics2 function to calculate additional metrics using predicted probabilities and saves the plot at the given path.

    The method returns a tuple containing the training accuracy and validation accuracy.
    """
    if simple:
        mlp = MLPClassifier(solver='sgd', random_state=42, learning_rate_init=0.01,
                            hidden_layer_sizes=(1024, 512))
        mlp.fit(x_train, y_train)
        y_train_pred = mlp.predict(x_train)
        y_val_pred = mlp.predict(x_test)
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_test, y_val_pred)
        y_val_prob = mlp.predict_proba(x_test)
        if plot_path:
            calculate_metrics2(y_test, y_val_pred, y_val_prob, plot_path=plot_path)

    else:
        mlp = MLPClassifier(max_iter=1, warm_start=True, solver='sgd', random_state=42, learning_rate_init=0.01, hidden_layer_sizes=(1024, 512))
        for epoch in range(NUM_EPOCHS):
            mlp.partial_fit(x_train, y_train, np.unique(y_train))
            y_train_pred = mlp.predict(x_train)
            y_val_pred = mlp.predict(x_test)
            train_acc = accuracy_score(y_train, y_train_pred)
            val_acc = accuracy_score(y_test, y_val_pred)
            train_loss = log_loss(y_train, mlp.predict_proba(x_train))
            val_loss = log_loss(y_test, mlp.predict_proba(x_test))
            print(
                f"Epoch {epoch + 1}/{NUM_EPOCHS} - train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

    return train_acc, val_acc



def fit_xgb_classifier(x_train, y_train, x_test, y_test, plot_path=None):
    """
    Fit a XGBoost classifier model using the training data.

    :param x_train: The training data features.
    :type x_train: numpy.ndarray

    :param y_train: The training data labels.
    :type y_train: numpy.ndarray

    :param x_test: The test data features.
    :type x_test: numpy.ndarray

    :param y_test: The test data labels.
    :type y_test: numpy.ndarray

    :param plot_path: The path to save the ROC curve plot. Default is None.
    :type plot_path: str, optional

    :return: The training accuracy score and validation accuracy score.
    :rtype: tuple
    """
    # Create a XGBClassifier model
    model = XGBClassifier()

    # Fit the model using the training data
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    y_test_prob = model.predict_proba(x_test)
    # Calculate accuracies
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_test, y_test_pred)
    # If plot_path is not None, compute the metrics and plot ROC curve
    if plot_path:
        calculate_metrics2(y_test, y_test_pred, y_test_prob, plot_path)


    return train_acc, val_acc


def train(classifier_type, train_file, val_file, train_dir, val_dir, plot_path=None, num_features=None):
    """
    Trains a classifier using the specified method.

    :param classifier_type: The type of classifier to use. Can be either "XGB" for XGBoost or any other value for MLP.
    :param train_file: The file path for the training data.
    :param val_file: The file path for the validation data.
    :param train_dir: The directory path for the training images.
    :param val_dir: The directory path for the validation images.
    :param plot_path: Optional. The file path to save a plot of the training and validation accuracy.

    :return: A tuple containing the training accuracy and validation accuracy.
    """
    NUM_CLASSES = 2

    only_dataset = None
    label_index = -1
    train_dataset = CustomDataset.ImageDataset(filepath=train_file, img_dir=train_dir,
                                 label_index=label_index, is_val=False, num_features=num_features)
    val_dataset = CustomDataset.ImageDataset(filepath=val_file, img_dir=val_dir,
                               label_index=label_index, is_val=True, num_features=num_features)

    x_train, y_train = train_dataset.get_features_labels()
    x_test, y_test = val_dataset.get_features_labels()
    y_train, y_test = np.ravel(y_train), np.ravel(y_test)
    print(f"Dataset size: {len(y_train)}, {len(y_test)}, {np.unique(y_train)}, {np.unique(y_test)}")
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    if classifier_type == "XGB":
        train_acc, val_acc = fit_xgb_classifier(x_train, y_train, x_test, y_test, plot_path=plot_path)
    else:
        train_acc, val_acc = fit_mlp_classifier(x_train, y_train, x_test, y_test, plot_path=plot_path)

    return train_acc, val_acc



def train_multiple_datasets():
    """
    Train Multiple Datasets

    This method trains multiple datasets using various models and model types. It iterates over the datasets, models, and model types to train and evaluate each combination. The results
    * are stored in a DataFrame and printed at the end.

    :return: None

    Parameters:
    None

    Usage:
    train_multiple_datasets()
    """
    datasets = ["Cervical", "breakhis-final"]
    models = [
        {"model_name": "vgg16", "features": 512},
        {"model_name": "vgg19", "features": 512},
        {"model_name": "inception", "features": 2048},
        {"model_name": "efficientnet-b0", "features": 1280},
        {"model_name": "efficientnet-b1", "features": 1280},
        {"model_name": "efficientnet-b2", "features": 1408},
        {"model_name": "mednet", "features": 1280},
    ]
    model_types = ["MLP", "XGB"]
    results = []

    for dataset in datasets:
        base_dir = f"{generic_dir}other_datasets/{dataset}/"
        train_dir = f"{base_dir}train/"
        val_dir = f"{base_dir}validation/"
        for model in models:
            train_file = f"{train_dir}{model['model_name']}_train_data.csv"
            val_file = f"{val_dir}{model['model_name']}_train_data.csv"
            for model_type in model_types:
                train_acc, val_acc = train(model_type, train_file, val_file, train_dir, val_dir, plot_path=None, num_features= model["features"])
                print(f"dataset: {dataset}, \tmodel: {model}, \tmodel_type: {model_type},\ttrain_acc: {train_acc:.4f}, \tval_acc: {val_acc:.4f}")
                results.append([dataset, model, model_type, train_acc, val_acc])
    df = pd.DataFrame(results, columns=["Dataset", "Features From", "Model type", "Train Acc", "Val Acc"])
    print(df)



def print_dataset_details():
    """
    Print dataset details.

    :return: None
    """
    datasets = [ "other_datasets/breakhis-final/40X/", "other_datasets/Cervical/"]
    label_indices = [-1, -1]
    num_classes = [2, 5]
    base_dir = f"{generic_dir}"
    models = ["MLP", "XGB"]
    for only_dataset, label_index, NUM_CLASSES in zip(datasets, label_indices, num_classes):
        for model in models:
            train_dir = f"{base_dir}{only_dataset}train/"
            val_dir = f"{base_dir}{only_dataset}validation/"
            train_file = f"{train_dir}mednet_train_data.csv"
            val_file = f"{val_dir}mednet_train_data.csv"
            print("Dataset: ", only_dataset)
            ds_str = only_dataset.replace('/', '_')
            tacc, vacc = train(
                classifier_type=model,
                train_file=train_file,
                val_file=val_file,
                train_dir=train_dir,
                val_dir=val_dir,
                num_features=50,
                plot_path=f"diagrams/{ds_str}_{model}.png"
            )
            print(f"Accuracies: {tacc}, {vacc}")


if __name__ == "__main__":
    train_multiple_datasets()