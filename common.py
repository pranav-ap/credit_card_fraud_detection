import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


"""
Skewness
"""


def fix_skewness_yeojohnson(df, skew_threshold=0.5):
    from scipy.stats import yeojohnson

    df_fixed = df.copy()

    numeric_columns = df_fixed.select_dtypes(include=['number']).columns

    # Apply Yeo-Johnson transformation
    for column in numeric_columns:
        skewness = df_fixed[column].skew()
        if abs(skewness) > skew_threshold:
            # print(f"Column '{column}' has skewness of {skewness:.2f}. Applying Yeo-Johnson transformation.")

            df_fixed[column], _ = yeojohnson(df_fixed[column])

    return df_fixed


"""
Missing Values
"""


def null_table(data):
    """
    A function which returns the number and percentage of null values in the given dataset.
    """
    indices = data.isnull().sum().index
    values = data.isnull().sum().values
    percentages = []

    for i in indices:
        percentages.append((data[i].isnull().sum() / data[i].shape[0]) * 100)

    d = {'Columns': indices,
         'Null Count': values,
         'Percentage': percentages}

    null_frame = pd.DataFrame(data=d)
    null_frame = null_frame.sort_values(
        by='Percentage',
        ascending=False
    ).reset_index(drop=True).query('Percentage > 0')

    return null_frame


def corr_among_numerics(df):
    f, ax = plt.subplots(figsize=(8, 6))

    corr_matrix = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        square=True,
        annot=True,
        fmt='.2f',
        annot_kws={'size': 14},
        cmap="BuPu"
    )

    plt.title('Correlation among Numerical Features', fontsize=16, pad=25)
    plt.xticks(rotation=25, fontsize=15)
    plt.yticks(rotation=25, fontsize=15)

    plt.show()


def plot_confusion_matrix(y_test, y_pred, cmap='Blues', text_size=22):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # Customize the display
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=cmap, ax=ax, colorbar=False)

    # Add colorbar with customized location and size
    cbar = fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=13)

    # Titles and labels
    plt.title('Confusion Matrix', fontsize=18, pad=30)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)

    # Customize tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Increase the size of numbers inside the matrix and remove unwanted formatting
    for text in ax.texts:
        text.set_fontsize(text_size)

    # Show the plot with a tighter layout
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curve(y_true, y_pred_prob):
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score

    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    average_precision = average_precision_score(y_true, y_pred_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="g", lw=2, label=f"Precision-Recall curve (area = {average_precision:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.show()


def plot_roc_curve(y_true, y_pred_prob):
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="g", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.show()
