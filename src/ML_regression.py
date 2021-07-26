


def ML_regression(data, labels, method="linear"):
    """Various ML regression models.

    Args:
        data ([array-like]): training data to train the model on
        labels ([array-like]): correspoding labels for the training data
        method ([str], optional): regression method to use (options: 'linear', 'lasso', 'ridge', 'elastic-net', 'random-forest'). Defaults to 'linear'.

    Returns:
        reg ([object]): returns a regression model trained on the given data and labels
    """

    if method == "linear":
        reg = LinearRegression().fit(data, labels)
    elif method == "lasso":
        reg = Lasso().fit(data, labels)
    elif method == "ridge":
        reg = Ridge().fit(data, labels)
    elif method == "elastic-net":
        reg = ElasticNet().fit(data, labels)
    elif method == "random-forest":
        reg = RandomForestRegressor().fit(data, labels)
    else:
        raise ValueError(
            "Regression {} is unknown. Please choose: 'linear', 'lasso', 'ridge', 'elastic-net', 'random-forest'".format(
                method
            )
        )

    return reg