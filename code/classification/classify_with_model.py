import argparse
import sys
import os
import random
import collections
import joblib
from typing import List
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score
from treeinterpreter import treeinterpreter as ti

#from logger import LOGGER

def build_cpt_dict():

    openwpm_resource_types = [
        "beacon",
        "csp_report",
        "font",
        "image",
        "imageset",
        "main_frame",
        "media",
        "object",
        "object_subrequest",
        "ping",
        "script",
        "stylesheet",
        "sub_frame",
        "web_manifest",
        "websocket",
        "xml_dtd",
        "xmlhttprequest",
        "xslt",
        "other",
    ]

    cpt_dict = dict((v,k) for k,v in enumerate(openwpm_resource_types))
    return cpt_dict 

def convert_cpt_old(cpt, cpt_dict):

    """
    Helper function to convert content_policy_type to old OpenWPM format.

    Args:
        cpt: Content policy type
        cpt_dict: Mapping of string to integer for cpt
    Returns:
        Corrected cpt format.
    """

    if cpt in cpt_dict:
        return cpt_dict[cpt]
    else:
        return cpt
        
def get_perc(num, den):

    """
    Helper function to get percentage value.

    Args:
        num: Numerator
        den: Denominator
    Returns:
        Percentage, rounded to 2 decimal places.
    """

    return str(round(num/den * 100, 2)) + "%"

def print_stats(report, result_dir, avg='macro avg', stats=['mean', 'std']):

    """
    Function to make classification stats report. This gives us the metrics over all folds.

    Args:
        report: Results of all the folds.
        result_dir: Output directory for results.
        avg: Type of average we want (macro).
        stats: Stats we want (mean/std).
    Returns:
        Nothing, writes to a file.
    """

    by_label = report.groupby('label').describe()
    fname = os.path.join(result_dir, "scores")
    with open(fname, "w") as f:
        for stat in stats:
            print(by_label.loc[avg].xs(stat, level=1))
            x = by_label.loc[avg].xs(stat, level=1)
            f.write(by_label.loc[avg].xs(stat, level=1).to_string())
            f.write("\n")

def report_feature_importance(feature_importances, result_dir):

    """
    Function to log feature importances to a file.

    Args:
        feature_importances: Feature importances.
        result_dir: Output directory for results.
    Returns:
        Nothing, writes to a file.
    """

    fname = os.path.join(result_dir, "featimp")
    with open(fname, "a") as f:
        f.write(feature_importances.to_string())
        f.write("\n")

def report_true_pred(y_true, y_pred, name, vid, i, result_dir):

    """
    Function to make truth/prediction output file, and confustion matrix file.

    Args:
        y_true: Truth values.
        y_pred: Predicted values.
        name: Classified resource URLs.
        vid: Visit IDs.
        i: Fold number.
        result_dir: Output directory.
    Returns:
        Nothing, writes to files.
    """

    fname = os.path.join(result_dir, "tp_%s" % str(i))
    with open(fname, "w") as f:
        for i in range(0, len(y_true)):
            f.write("%s |$| %s |$| %s |$| %s\n" %(y_true[i], y_pred[i], name[i], vid[i]))

    fname = os.path.join(result_dir, "confusion_matrix")
    with open(fname, "a") as f:
        f.write(np.array_str(confusion_matrix(y_true, y_pred, labels=[True, False])) + "\n\n")

def describe_classif_reports(results, result_dir):

    """
    Function to make classification stats report over all folds.

    Args:
        results: Results of classification
        result_dir: Output directory
    Returns:
        all_folds: DataFrame of results

    This functions does the following:

    1. Obtains the classification metrics for each fold.
    """

    true_vectors, pred_vectors, name_vectors, vid_vectors = [r[0] for r in results], [r[1] for r in results], [r[2] for r in results], [r[3] for r in results]
    fname = os.path.join(result_dir, "scores")

    all_folds = pd.DataFrame(columns=['label', 'fold', 'precision', 'recall', 'f1-score', 'support'])
    for i, (y_true, y_pred, name, vid) in enumerate(zip(true_vectors, pred_vectors, name_vectors, vid_vectors)):
        report_true_pred(y_true, y_pred, name, vid, i, result_dir)
        output = classification_report(y_true, y_pred)
        with open(fname, "a") as f:
            f.write(output)
            f.write("\n\n")
    return all_folds


def classify_with_model(clf, test, result_dir, feature_list):

    """
    Function to perform classification.

    Args:
        clf: Trained model.
        test: Test data.
        result_dir: Output folder for results.
        feature_list: List of features to include. (not using in new version)
    Returns:
        list(test_mani.label): Truth labels of test data.
        list(y_pred): Predicted labels of test data.
        list(test_mani.name): URLs of test data.
        list(test_mani.visit_id): Visit IDs of test data.
    """

    test_mani = test.copy()
    fields_to_remove = ['visit_id', 'name', 'label']
    df_feature_test = test_mani.drop(fields_to_remove, axis=1)
    df_feature_test = df_feature_test.to_numpy()
    
    y_pred = clf.predict(df_feature_test)
    
    acc = accuracy_score(test_mani.label, y_pred)
    prec = precision_score(test_mani.label, y_pred, pos_label=True)
    rec = recall_score(test_mani.label, y_pred, pos_label=True)

    fname = os.path.join(result_dir, "accuracy")
    with open(fname, "a") as f:
        f.write("\nAccuracy score: " + str(round(acc*100, 3)) + "%" + "\n")
        f.write("Precision score: " + str(round(prec*100, 3)) + "%" + "\n")
        f.write("Recall score: " + str(round(rec*100, 3)) + "%" +  "\n")

    return list(test_mani.label), list(y_pred), list(test_mani.name), list(test_mani.visit_id)


def pipeline(model_file, feature_file, label_file, result_dir):

    """
    Function to run classification pipeline.

    Args:
      model_file: Joblib file of trained model.
      feature_file: CSV file of features from feature extraction process.
      label_file: CSV file of labels from labelling process.
      result_dir: Output folder for results.
    Returns:
      Nothing, creates a result directory with all the results.
    """

    df_features = pd.read_csv(feature_file)
    df_features.dropna(axis=1, how="all", inplace=True)
    df_features.dropna(how="any", inplace=True)
    
    if 'content_policy_type' in df_features.columns:
        cpt_dict = build_cpt_dict()
        df_features['content_policy_type'] = \
            df_features['content_policy_type'].apply(convert_cpt_old, cpt_dict=cpt_dict)

    df_labels = pd.read_csv(label_file)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    clf = joblib.load(model_file)
    feature_list = []

    df_labelled = df_features.merge(df_labels[['visit_id', 'name', 'label']], on=['visit_id', 'name'])
    df_labelled = df_labelled[df_labelled['label'] != "Error"]

    result = classify_with_model(clf, df_labelled, result_dir, feature_list)
    results = [result]
    report = describe_classif_reports(results, result_dir)

def main(program: str, args: List[str]):

    parser = argparse.ArgumentParser(prog=program, description="Run the WebGraph classification pipeline.")

    parser.add_argument(
        "--model",
        type=str,
        help="Model joblib file.",
        default="model.joblib"
    )
    parser.add_argument(
        "--features",
        type=str,
        help="Features CSV file.",
        default="features.csv"
    )
    parser.add_argument(
        "--labels",
        type=str,
        help="Labels CSV file.",
        default="labels.csv"
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Directory to output the results.",
        default="results"
    )

    ns = parser.parse_args(args)
    pipeline(ns.model, ns.features, ns.labels, ns.out)


if __name__ == "__main__":

    main(sys.argv[0], sys.argv[1:])

