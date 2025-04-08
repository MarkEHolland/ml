"""
Summaries of sklearn.pipeline
"""

from pandas.api.types import CategoricalDtype
from typing import Optional, Tuple, Union, List
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from loguru import logger
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from IPython.display import display, HTML
from sklearn.metrics import (
#    RocCurveDisplay,
#    plot_roc_curve,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    RocCurveDisplay,
    precision_recall_curve
)
import shap
from sklearn.pipeline import Pipeline


def model_summary(
    model: Pipeline,
    X_test_df: pd.DataFrame,
    y_test: [],
    dataset_name: str
) -> None:
    """
    Model report with test data:
    1) Target barplot (top-left)
    2) Lift curve (top-right)
    3) Receiver Operator Curve (ROC) (bottom-right)
    4) Confusion matrix (bottom-left)
        Accuracy = what proportion did we get right
        Precision = what proportion of predicted 1's are correct
        Recall = what % of actual 1's did the model pick out
        Type 1 error (false positive) - actual 1, prediction 0
        Type 2 error (false negative) - actual 0, prediction 1
    """

    sns.set(font_scale=1)
    X = X_test_df
    y = pd.DataFrame(y_test)
    preds = model.predict(X)
    prob_preds = model.predict_proba(X)[:, 1]

    fig = plt.figure(figsize=(18, 8), constrained_layout=True)
    spec = gridspec.GridSpec(ncols=4, nrows=2, figure=fig, hspace=0.1, wspace=0.1)
    ax1 = fig.add_subplot(spec[0, 0])  # Barplot: first row and first column
    ax2 = fig.add_subplot(spec[0, -3:-2])  # Barplot: first row and second column
    ax3 = fig.add_subplot(spec[0, -2:])  # Lift Curve: first rows and last two columns
    ax4 = fig.add_subplot(spec[-1, 0]) # Classification matrix: bottom row and first columns
    ax5 = fig.add_subplot(spec[-1, -3:-2])  # Confusion matrix: bottom row and second columns
    ax6 = fig.add_subplot(spec[-1, -2:]) # ROC Curve: bottom row and last two columns

    # x and y label sizes
    ax1.yaxis.set_tick_params(labelsize='large')
    ax1.xaxis.set_tick_params(labelsize='large') 
    ax2.yaxis.set_tick_params(labelsize='large')
    ax2.xaxis.set_tick_params(labelsize='large') 
    ax3.yaxis.set_tick_params(labelsize='large')
    ax3.xaxis.set_tick_params(labelsize='large') 
    ax4.yaxis.set_tick_params(labelsize='large')
    ax4.xaxis.set_tick_params(labelsize='large') 
    ax5.yaxis.set_tick_params(labelsize='large')
    ax5.xaxis.set_tick_params(labelsize='large')   
    ax6.yaxis.set_tick_params(labelsize='large')
    ax6.xaxis.set_tick_params(labelsize='large')  
    
    # Plot 1: Target Barplot
    sns.set(font_scale=1.0)
    g= sns.barplot(x=y_test,y=y_test, estimator=lambda x: len(x) / len(y_test), ax=ax1, palette=sns.color_palette())
    #Anotating the graph
    for p in g.patches:
        width, height = p.get_width(), p.get_height()
        x_pos, y_pos = p.get_xy() 
        g.text(x_pos+width/2, 
               y_pos+height, 
               '{:.1%}'.format(height), 
               horizontalalignment='center',fontsize=12)
    
    #Setting the labels
    ax1.set_xlabel('target (actual)', fontsize=12)
    ax1.set_ylabel('% of obs', fontsize=12)
    ax1.set_title(f'Target (#obs = {len(y_test):.0f})', fontsize=16)

    
    # Plot 2: Classification Results Histogram
    catbool_dtype = CategoricalDtype(categories=['0', '1'], ordered=True)    
    plot_df = pd.DataFrame({'target':y_test,'prediction':preds})
    plot_df = plot_df.astype(str).astype(catbool_dtype)
    sns.histplot(data=plot_df, x=plot_df.prediction, hue=plot_df.target, stat="count",element="bars",ax=ax2, cumulative=False)
    ax2.axhline(y_test.mean(), 0,1,color='gray',linestyle='--',label='average',linewidth=1)
    ax2.set_ylabel('# target (actual)', fontsize=12)
    ax2.set_xlabel('prediction', fontsize=12)
    ax2.set_title("prediction v target (actual)",fontsize=16)        

    
    # Plot 3: Plot Lift curve
    percentiles = [0] + list(np.percentile(prob_preds, np.arange(10, 110, 10)))

    step = 0.1
    # Define an auxiliar dataframe to plot the curve
    aux_lift = pd.DataFrame()
    # Create a real and predicted column for our new DataFrame and assign values
    aux_lift["real"] = y
    aux_lift["predicted"] = prob_preds
    aux_lift["predicted_bin"] = preds
    aux_lift["decile"] = np.floor(aux_lift["predicted"] * 10) / 10
    # Order the values for the predicted probability column:
    aux_lift.sort_values("predicted", ascending=False, inplace=True)

    # Create the values that will go into the X axis of our plot
    x_val = np.arange(step, 1 + step, step)
    # Calculate the ratio of ones in our data
    ratio_ones = aux_lift["real"].sum() / len(aux_lift)
    # Calculate the total number of predicted conversion
    total_pred_conversions = aux_lift["real"].sum()
    # Create empty vectors with the values that will go on the Y axis our our plot
    y_v = []
    cum_conv = []
    avg_conv = []

    # Calculate for each x value its correspondent y value
    for x in range(10):
        data_here = aux_lift[
            (aux_lift["predicted"] >= percentiles[x])
            & (aux_lift["predicted"] <= percentiles[x + 1])
        ]
        ratio_ones_here = data_here["real"].sum() / len(data_here)
        y_v.append(ratio_ones_here / ratio_ones)
        cum_conv.append(
            aux_lift[aux_lift["predicted"] >= percentiles[x]]["real"].sum()
            / total_pred_conversions
        )
        avg_conv.append(ratio_ones_here)

    # Plot the figure
    ax3_1 = ax3.twinx()
    ax3.bar(x_val, y_v[::-1], width=0.08, color=sns.color_palette()[9], edgecolor =None)
    ax3_1.plot(x_val, cum_conv[::-1], linewidth=1, markersize=5,color="gray",linestyle="dashed")
    ax3.set_ylim([0, ax3.get_ylim()[1] + 0.2])

    rects = ax3.patches

    # Make some labels.
    labels = ["{:,.1%}".format(c) for c in avg_conv[::-1]]

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax3.text(
            rect.get_x() + rect.get_width() / 2,
            height + 0.1,
            label,
            ha="center",
            va="bottom",
        )

    ax3.set_xticks(x_val, labels=[9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    ax3.set_xlabel("Lift and % purity of target in each decile", fontsize=12)
    ax3.set_ylabel("Lift & % purity in each decile", fontsize=12)
    ax3.set_title("Lift, % purity & Cumulated target", fontsize=16)
    ax3_1.set_ylim([0, 1.05])
    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax3_1.set_yticks(ticks)
    ax3_1.set_yticklabels(["{0:.0%}".format(t) for t in ticks])
    ax3_1.set_ylabel("------ Cumulated target %", fontsize=12)

    # Plot 4: classification report
    # see https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report
    clf_report = classification_report(y, preds, output_dict=True)
    keys_to_plot = [key for key in clf_report.keys() if key not in ('accuracy', 'macro avg', 'weighted avg')]
    df = pd.DataFrame(clf_report, columns=keys_to_plot).T
    #the following line ensures that dataframe are sorted from the majority classes to the minority classes
    df.sort_values(by=['support'], inplace=True) 

    #first, let's plot the heatmap by masking the 'support' column
    rows, cols = df.shape
    mask = np.zeros(df.shape)
    mask[:,cols-1] = True

    ax4_1 = ax4.twinx()
    # plot metrics
    ax4_1 = sns.heatmap(df, mask=mask, annot=True, cmap="YlGn", fmt='.3g',
            vmin=0.0,
            vmax=1.0,
            linewidths=2, linecolor='white',ax=ax4
                    )

    #then, let's add the support column by normalizing the colors in this column
    mask = np.zeros(df.shape)
    mask[:,:cols-1] = True    

    # plot support
    ax4_2 = sns.heatmap(df, mask=mask, annot=True, cmap="YlGn", cbar=False, ax=None,
            linewidths=2, linecolor='white', fmt='.0f',
            vmin=df['support'].min(),
            vmax=df['support'].sum(),         
            norm=mpl.colors.Normalize(vmin=df['support'].min(),
                                      vmax=df['support'].sum())
                    ) 

    ax4.set_title(f"Classification Report: accuracy {clf_report['accuracy']:.1%}", fontsize=16)
    ax4.set_xlabel("Metric", fontsize=12)
    ax4.set_ylabel("target (actual)", fontsize=12)
    
    
    # Plot 5: Confusion Matrix
    #Create confusion matrix
    cf_matrix = confusion_matrix(y, preds)
    group_names = ['True Neg','False Pos (type I)','False Neg (Type II)','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ['{0:.1%}'.format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt="", cmap="Blues", ax=ax5)
    ax5.set_title("Confusion Matrix",fontsize=16)    
    ax5.set_xlabel('prediction', fontsize=12)
    ax5.set_ylabel('target (actual)', fontsize=12)

    # Plot 6: Plot ROC curve
    RocCurveDisplay.from_estimator(model, X, y, ax=ax6)
    ax6.set_title("ROC Curve",fontsize=16)

    plt.suptitle(f"{dataset_name} data",fontsize=18)
    plt.show()
    
    
def shap_summary(
    model: Pipeline,
    X_test_df: pd.DataFrame,
    y_test: [],
    dataset_name: str
) -> None:
    """ """

    X = X_test_df
    y = pd.DataFrame(y_test)
    preds = model.predict(X)
    prob_preds = model.predict_proba(X)[:, 1]