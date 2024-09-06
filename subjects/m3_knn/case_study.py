import pandas as pd
import numpy as np
from math import pi
from dataset_man import DatasetManager
from sklearn.neighbors import KDTree
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def main():
    # Using KNN with KDTree algorithm to solve unsupervised issue
    # KDTree for fast generalized N-point problems
    dm = DatasetManager()
    print(dm.data_list)

    df = dm.load_dataset('m3.knn.20240828.votes_skills_anon', as_dataframe=True)
    df.set_index(['Unnamed: 0'], inplace=True)
    df.index.name = None
    print(df.head())
    grouped_skills_df = df.groupby(['univid', 'response'])['response'].count()
    print(grouped_skills_df.head(10))
    print(type(grouped_skills_df.index))
    print(grouped_skills_df.index[:10])
    grouped_skills_df = grouped_skills_df.rename('value').reset_index()
    print(grouped_skills_df.head(10))
    print(type(grouped_skills_df.index))
    univ_skills_df = grouped_skills_df.pivot_table(values='value', columns='response', index=['univid'])
    print(type(univ_skills_df))
    print(univ_skills_df.info())
    print(univ_skills_df.isna().sum())
    print(univ_skills_df.head())

    # Standardization
    univ_skills_std_df = univ_skills_df.div(univ_skills_df.sum(axis=1), axis=0)
    print(univ_skills_std_df.head())

    # Process NA data
    univ_skills_std_df.fillna(0, inplace=True )

    # Inspect data
    sns.boxplot(univ_skills_std_df)
    plt.show()

    sns.kdeplot(univ_skills_std_df)
    plt.show()

    # Model instantiation
    tree = KDTree(univ_skills_std_df, metric='euclidean')
    print('valid metrics: ', tree.valid_metrics)
    print('query data: \n', univ_skills_std_df[9:10])
    dist, ind = tree.query(univ_skills_std_df[9:10], k=5)
    print('inference indices: ', ind.tolist()[0])
    print('dist:\n', dist)
    data_u = univ_skills_std_df.iloc[ind.tolist()[0]]
    print(data_u.head())
    print(data_u["Academic excellence"])

    # Radar visualization
    offDisplay = False
    if offDisplay: sys.exit()
    # Data

    categories = data_u.columns.values.tolist()
    values = [4, 3, 2, 5, 4]

    # Number of variables
    num_vars = len(categories)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The radar chart requires a complete loop, so append the start value to the end
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2)

    # Labels for each category
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    plt.show()

    def plot_radar(df):
        # Number of entities
        num_entities = len(df.columns)

        # Number of categories
        num_vars = len(df.index)

        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # Draw one axe per variable and add labels
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], df.index)

        # Plot each entity
        for column in df.columns:
            values = df[column].tolist()
            values += values[:1]  # Close the loop
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=column)
            ax.fill(angles, values, alpha=0.25)

        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

        plt.show()

    plot_radar(data_u)

    def plot_radar_v2(df):
        # Number of entities
        num_entities = len(df.columns)

        # Number of categories
        num_vars = len(df.index)

        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # Draw one axe per variable and add labels
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], df.index)

        # Plot each entity
        for column in df.columns:
            values = df[column].tolist()
            values += values[:1]  # Close the loop
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=column)
            ax.fill(angles, values, alpha=0.25)

        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

        plt.show()

    df = univ_skills_std_df.iloc[[2, 3, 5]]
    plot_radar_v2(df)

if __name__ == "__main__":
    main()
