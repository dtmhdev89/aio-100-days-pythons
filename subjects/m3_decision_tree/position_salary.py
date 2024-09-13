import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor, export_graphviz

from matplotlib import pyplot as plt

from dataset_man import dataset_manager

def main():
    dm = dataset_manager
    
    pos_sal_data = dm.load_dataset('m3.decision.tree.regression.20240906.Position_Salaries')
    df = pd.DataFrame(pos_sal_data)
    print(df.info())
    print(df.columns)
    print(df.isna().sum())

    X = df.loc[:, ['Level']].values
    print(type(X))
    print(X.shape)
    print(X)

    y = df.loc[:, ['Salary']].values
    print(y.shape)

    pipeline_1 = Pipeline([
        ('ds_rg', DecisionTreeRegressor())
    ])

    pipeline_1.set_params(ds_rg__random_state=0, ds_rg__max_depth=3)
    regressor_max_depth_three = pipeline_1.named_steps['ds_rg']
    print(regressor_max_depth_three)

    regressor_max_depth_three.fit(X, y)

    pipeline_2 = Pipeline([
        ('ds_rg', DecisionTreeRegressor())
    ])

    pipeline_2.set_params(ds_rg__max_depth=None, ds_rg__random_state=0, ds_rg__min_samples_leaf=4)
    regressor_min_samples_leaf_fourth = pipeline_2.named_steps['ds_rg']
    print(regressor_min_samples_leaf_fourth)

    regressor_min_samples_leaf_fourth.fit(X, y)

    fig, ax = plt.subplots(figsize=(12, 8))
    plot_data = tree.plot_tree(regressor_max_depth_three, ax=ax, feature_names = ["Level"], filled=True)
    print(plot_data)
    plt.show()

    export_graphviz(regressor_max_depth_three, out_file ='tree.dot',
               feature_names =["Level"])
    
    plt.figure()
    plt.scatter(X, y, marker= "x", color = 'red', label = "Data")
    plt.plot(X, regressor_max_depth_three.predict(X), color = 'blue', label = "max depth = 3")
    plt.plot(X, regressor_min_samples_leaf_fourth.predict(X), marker= "D", color = 'green', label = "min sample leaf = 4")
    plt.title('Check It (Regression Model)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
