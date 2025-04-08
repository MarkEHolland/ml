# overview of methods: https://practicalml.net/Detecting-data-drift/

import numpy as np
import pandas as pd
import copy
import json
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import CatBoostEncoder
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde, ks_2samp, chisquare, wasserstein_distance
from scipy.special import rel_entr
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (r2_score, mean_absolute_error, precision_score,
                             recall_score, accuracy_score, f1_score,
                             roc_auc_score)
from sklearn.utils import shuffle
import logging
logger = logging.getLogger()
sns.set_theme(style="ticks")

class DataDriftDetector:
    # source: https://github.com/kelvnt/data-drift-detector/blob/master/data_drift_detector/data_drift_detector.py
    """Compare differences between 2 datasets
    DataDriftDetector creates useful methods to compare 2 datasets,
    created to allow ease of measuring the fidelity between 2 datasets.
    
    Methods
    ----
    calculate_drift:
        Calculates the distribution distance for each column between the
        datasets
    plot_numeric_to_numeric:
        Creates a pairplot between the 2 datasets
    plot_categorical_to_numeric:
        Creates a pairgrid violin plot between the 2 datasets
    plot_categorical:
        Creates a proportion histogram between the 2 datasets for categorical
        columns
    compare_ml_efficacy:
        Compares the ML efficacy of a model built between the 2 datasets
    Args
    ----
    df_prior: <pandas.DataFrame>
        Pandas dataframe of the prior dataset. In practice, this would be the
        dataset used to train a live model
    df_post: <pandas.DataFrame>
        Pandas dataframe of the post dataset. In practice, this would be the
        current dataset that's flowing into a live model
    categorical_columns: <list of str>
        A list of categorical columns in the dataset, will be determined by
        column types if not provided
    numeric_columns: <list of str>
        A list of numeric columns in the dataset, will be determined by
        column types if not provided
    """
    def __init__(self,
                 df_prior,
                 df_post,
                 categorical_columns=None,
                 numeric_columns=None):
        assert isinstance(df_prior, pd.DataFrame),\
            "df_prior should be a pandas dataframe"
        assert isinstance(df_post, pd.DataFrame),\
            "df_post should be a pandas dataframe"
        assert sorted(df_prior.columns) == sorted(df_post.columns),\
            "df_prior and df_post should have the same column names"
        assert all(df_prior.dtypes.sort_index() == df_post.dtypes.sort_index()),\
            "df_prior and df_post should have the same column types"
        assert isinstance(categorical_columns, (list, type(None))),\
            "categorical_columns should be of type list"
        assert isinstance(numeric_columns, (list, type(None))),\
            "numeric_columns should be of type list"

        df_prior_ = df_prior.copy()
        df_post_ = df_post.copy()

        if categorical_columns is None:
            categorical_columns = (
                [c for c in df_prior_.columns if
                df_prior_.dtypes[c] == 'object']
            )
            logger.info(
                "Identified categorical column(s): {}".format(
                ", ".join(categorical_columns))
            )

        df_prior_[categorical_columns] = (
            df_prior_[categorical_columns].astype(str)
        )
        df_post_[categorical_columns] = (
            df_post_[categorical_columns].astype(str)
        )

        if numeric_columns is None:
            num_types = ['float64','float32','int32','int64','uint8']
            numeric_columns = (
                [c for c in df_prior_.columns if
                 df_prior_.dtypes[c] in num_types]
            )
            logger.info("Identified numeric column(s): {}".format(
                ", ".join(numeric_columns))
            )

        df_prior_[numeric_columns] = df_prior_[numeric_columns].astype(float)
        df_post_[numeric_columns] = df_post_[numeric_columns].astype(float)

        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns

        self.df_prior = df_prior_
        self.df_post = df_post_[df_prior_.columns]


    def calculate_drift(self, steps=100, n_bins=10):
        """Calculates metrics and test of similarity between the 2 datasets
        For categorical columns, the probability of each category will be
        computed separately for `df_prior` and `df_post`, and the distance 
        between the 2 probability arrays will be computed. 
        For numeric columns, the values will first be fitted into a gaussian KDE
        separately for `df_prior` and `df_post`, and a probability array
        will be sampled from them
        
        Args
        ----
        steps: int
            Number of steps to take to sample for the fitted KDE for numeric
            columns. Defaults to 10.
        n_bins: int
            Number of buckets for binning in probability stability index. Defaults to 10.
        Returns
        ----
        Dictionary of results
        """
        
        def _psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
            """Calculate PSI metric for two arrays.
            source adpated from: https://www.kaggle.com/code/podsyp/population-stability-index

            Parameters
            ----------
                expected : list-like
                    Array of expected values
                actual : list-like
                    Array of actual values
                bucket_type : str
                    Binning strategy. Accepts two options: 'bins' and 'quantiles'. Defaults to 'bins'.
                    'bins': input arrays are splitted into bins with equal
                        and fixed steps based on 'expected' array
                n_bins : int
                    Number of buckets for binning. Defaults to 10.

            Returns
            -------
                A single float number
            """
            breakpoints = np.arange(0, n_bins + 1) / (n_bins) * 100
            breakpoints = np.histogram(expected, n_bins)[1]

            # Calculate frequencies
            expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
            actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
            # Clip freaquencies to avoid zero division
            expected_percents = np.clip(expected_percents, a_min=0.0001, a_max=None)
            actual_percents = np.clip(actual_percents, a_min=0.0001, a_max=None)
            # Calculate PSI
            psi_value = (expected_percents - actual_percents) * np.log(expected_percents / actual_percents)
            psi_value = sum(psi_value)

            return psi_value
        
        # calculate all metrics  
        cat_res = {}
        num_res = {}

        for col in self.categorical_columns:
            # to ensure similar order, concat before computing probability
            col_prior = self.df_prior[col].to_frame()
            col_post = self.df_post[col].to_frame()
            col_prior['_source'] = 'prior'
            col_post['_source'] = 'post'

            col_ = pd.concat([col_prior, col_post], ignore_index=True)

            # aggregate and convert to probability array
            arr = (col_.groupby([col, '_source'])
                       .size()
                       .to_frame()
                       .reset_index()
                       .pivot(index=col, columns='_source')
                       .droplevel(0, axis=1)
                       .pipe(lambda df: df.loc[df.sum(axis=1).sort_values(ascending=False).index, :])
                  )

            arr_ = arr.div(arr.sum(axis=0),axis=1)
            arr_.fillna(0, inplace=True)

            # calculate statistical distances
            kl_post_prior = sum(rel_entr(arr_['post'].to_numpy(), arr_['prior'].to_numpy()))
            kl_prior_post = sum(rel_entr(arr_['prior'].to_numpy(), arr_['post'].to_numpy()))

            jsdistance = jensenshannon(arr_['prior'].to_numpy(), arr_['post'].to_numpy())
            jsdivergence = jsdistance ** 2

            wd = wasserstein_distance(arr_['prior'].to_numpy(), arr_['post'].to_numpy())

            arr['post'] = arr['post'] / arr['post'].sum() * arr['prior'].sum()
            arr.fillna(0, inplace=True)
            
            # calculate test of similarity
            cs_test = chisquare(arr['post'].to_numpy(), arr['prior'].to_numpy())
            
            # calculate l-infinity of probabilities (i.e. max distance of probability difference)
            l_infinity = max(abs(arr_['prior'] - arr_['post']))
 
            # calculate psi
            psi = _psi(expected=arr['prior'], actual=arr['post'] )

            # for simplicity some statistics are not returned
            cat_res.update({
                col: {
#                    'chi_square_test_statistic': cs_test[0],
                    'chi_square_test_p_value': round(cs_test[1],3),
#                    'kl_divergence_post_given_prior': kl_post_prior,
#                    'kl_divergence_prior_given_post': kl_prior_post,
                    'jensen_shannon_divergence': jsdivergence,
#                    'wasserstein_distance': wd,
                    'l_infinity_norm': l_infinity
#                    'population_stability_index': psi
                }
            })

        for col in self.numeric_columns:
            # fit gaussian_kde
            col_prior = self.df_prior[col].dropna()
            col_post = self.df_post[col].dropna()
            kde_prior = gaussian_kde(col_prior)
            kde_post = gaussian_kde(col_post)

            # get range of values
            min_ = min(col_prior.min(), col_post.min())
            max_ = max(col_prior.max(), col_post.max())
            range_ = np.linspace(start=min_, stop=max_, num=steps)

            # sample range from KDE
            arr_prior_ = kde_prior.evaluate(range_)
            arr_post_ = kde_post.evaluate(range_)

            arr_prior = arr_prior_ / np.sum(arr_prior_)
            arr_post = arr_post_ / np.sum(arr_post_)

            # calculate statistical distances
            jsdistance = jensenshannon(arr_prior, arr_post)
            jsdivergence = jsdistance ** 2
                   
            wd = wasserstein_distance(arr_prior, arr_post)
            
            # calculate test of similarity
            #print(list(arr_prior),list(arr_post))
            ks_test = ks_2samp(arr_prior, arr_post)

            # calculate psi
            psi = _psi(expected=arr_prior, actual=arr_post )
            
            num_res.update({
                col: {
#                    'ks_2sample_test_statistic': ks_test[0],
                    'ks_2sample_test_p_value': ks_test[1],
                    'jensen_shannon_divergence': jsdivergence,
#                    'wasserstein_distance': wd,
#                    'population_stability_index': psi                    
                }
            })

        return {'categorical': dict(cat_res),
                'numerical': dict(num_res)}


    def plot_categorical_to_numeric(self,
                                    plot_categorical_columns=None,
                                    plot_numeric_columns=None,
                                    categorical_on_y_axis=True,
                                    grid_kws={'height':5},
                                    plot_kws={}):
        """Plots charts to compare categorical to numeric columns pairwise.
        Plots a pairgrid violin plot of categorical columns to numeric
        columns, split and colored by the source of datasets
        
        Args
        ----
        plot_categorical_columns: <list of str>
            List of categorical columns to plot, uses all if no specified
        plot_numeric_columns: <list of str>
            List of numeric columns to plot, uses all if not specified
        categorical_on_y_axis: <boolean>
            Determines layout of resulting image - if True, categorical
            columns will be on the y axis
        grid_kws: <dict>
            arguments to pass into the pair grid plot
        plot_kws: <dict>
            Arguments to pass into the violin plot
        Returns
        ----
        Resulting plot
        """
        assert isinstance(plot_categorical_columns, (list, type(None))),\
            "plot_categorical_columns should be of type list"
        assert isinstance(plot_numeric_columns, (list, type(None))),\
            "plot_numeric_columns should be of type list"
        assert isinstance(categorical_on_y_axis, bool),\
            "categorical_on_y_axis should be a boolean value"

        df_prior = self.df_prior.copy()
        df_post = self.df_post.copy()

        col_nunique = df_prior.nunique()

        if plot_categorical_columns is None:
            plot_categorical_columns = (
                [col for col in col_nunique.index if
                 (col_nunique[col] <= 20) & (col in self.categorical_columns)]
            )

        if plot_numeric_columns is None:
            plot_numeric_columns = self.numeric_columns

        df_prior["_source"] = "Prior"
        df_post["_source"] = "Post"

        plot_df = pd.concat([df_prior, df_post])
        
        msg = (
            "Plotting the following categorical column(s): " +
            ", ".join(plot_categorical_columns) +
            "\nAgainst the following numeric column(s):" +
            ", ".join(plot_numeric_columns) +
            "\nCategorical columns with high cardinality (>20 unique values)" +
            " are not plotted."
        )

        logger.info(msg)

        # violinplot does not treat numeric string cols as string - error
        # sln: added a whitespace to ensure it is read as a string
        plot_df[plot_categorical_columns] = (
            plot_df[plot_categorical_columns].astype(str) + " "
        )

        if categorical_on_y_axis:
            y_cols = plot_categorical_columns
            x_cols = plot_numeric_columns
        else:
            y_cols = plot_numeric_columns
            x_cols = plot_categorical_columns

        g = sns.PairGrid(data=plot_df,
                         x_vars=x_cols,
                         y_vars=y_cols,
                         hue='_source',
                         hue_kws={'split':True},
                         **grid_kws)

        g.map(sns.violinplot,
              hue=plot_df['_source'],
              split=True,
              **plot_kws)

        g.add_legend()
        g.fig.suptitle("categorical v numeric feature distributions",y=1.01)
        g.fig.show()
        
        return g


    def plot_numeric_to_numeric(self,
                                kind='scatter', #'scatter',
                                diag_kind='kde',
                                plot_kws=None,
                                grid_kws=None,
                                diag_kws={'common_norm':False},
                                plot_numeric_columns=None,
                                **kwargs):
        """Plots charts to compare numeric columns pairwise.
        Plots a pairplot (from seaborn) of numeric columns, with a distribution
        plot on the diagonal and a scatter plot for all other charts
        Args
        ----
        plot_numeric_columns: <list of str>
            List of numeric columns to plot, uses all if not specified
        kind: <str>
            Plot kind for the pair plot
        diag_kind: <str>
            Plot kind for the diagonal plots
        plot_kws: <dict>
            Arguments for both the diagonal and grid plots
        grid_kws: <dict>
            Arguments for the grid plots
        diag_kws: <dict>
            Arguments for the diagonal plots
        Returns
        ----
        Resulting plot
        """
        assert isinstance(plot_numeric_columns, (list, type(None))),\
            "plot_numeric_columns should be of type list"

        if plot_numeric_columns is None:
            plot_numeric_columns = self.numeric_columns

        df_prior = self.df_prior[plot_numeric_columns].copy()
        df_post = self.df_post[plot_numeric_columns].copy()

        df_prior['_source'] = "Prior"
        df_post['_source'] = "Post"

        plot_df = pd.concat([df_prior, df_post])
        plot_df.reset_index(drop=True, inplace=True)

        logger.info(
            "Plotting the following numeric column(s): {}".format(
            ", ".join(plot_numeric_columns))
        )

        g = sns.pairplot(data=plot_df,
                         kind=kind,
                         diag_kind=diag_kind,
                         hue='_source',
                         plot_kws=plot_kws,
                         diag_kws=diag_kws,
                         grid_kws=grid_kws,
                         **kwargs)

        g.fig.suptitle("numeric feature distributions",y=1.01)
        g.fig.show()
        
        return g


    def plot_categorical(self, plot_categorical_columns=None, **kwargs):
        """Plot histograms to compare categorical columns
        Args
        ----
        plot_categorical_columns: <list of str>
            List of categorical columns to plot, uses all if no specified
        Returns
        ----
        Resulting plot
        """
        assert isinstance(plot_categorical_columns, (list, type(None))),\
            "plot_categorical_columns should be of type list"

        col_nunique = self.df_prior.nunique()
        if plot_categorical_columns is None:
            plot_categorical_columns = (
                [col for col in col_nunique.index if
                 (col_nunique[col] <= 20) & (col in self.categorical_columns)]
            )

        logger.info(
            "Plotting the following categorical column(s): {}".format(
            ", ".join(plot_categorical_columns))
        )

        num_cat = len(plot_categorical_columns)
        nrows = (num_cat // 4) + 1
        ncols = min(num_cat,4)
        # col = index % ncols
        
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows))
                
        for index, val in enumerate(plot_categorical_columns,start=0):
            
            #if len(plot_categorical_columns) == 1:
            #    _ax = ax
            #elif len(plot_categorical_columns) > 1:
            #    _ax = ax[index]

            row = (index // ncols)
            col = index % ncols
            
            plt.subplot(nrows,ncols,index+1)    
                
            _p1 = (self.df_prior[val].value_counts(normalize=True)
                                     .rename("Proportion")
                                     .sort_index()
                                     .reset_index())
            _p2 = (self.df_post[val].value_counts(normalize=True)
                                    .rename("Proportion")
                                    .sort_index()
                                    .reset_index())
            _p1['_source'] = 'Prior'
            _p2['_source'] = 'Post'
            _p = pd.concat([_p1, _p2])

            sns.barplot(x="index",
                        y="Proportion",
                        hue="_source",
                        data=_p,
                        ax=axs[row,col],
                        **kwargs)
            plt.xlabel(f"{val}")
        plt.suptitle("categorical feature distributions")
        plt.tight_layout()
        plt.close(fig)
        plt.show()

        return fig

