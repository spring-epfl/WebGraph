import argparse
import os
import random
import collections
import joblib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score
from treeinterpreter import treeinterpreter as ti

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
    Function to make truth/prediction output file.

    Args:
        y_true: Truth values.
        y_pred: Predicted values.
        name: Classified resource URLs.
        vid: Visit IDs.
        i: Fold number.
        result_dir: Output directory.
    Returns:
        Nothing, writes to a file.

    This functions does the following:

    1. Obtains the classification metrics for each fold.
    """

    fname = os.path.join(result_dir, "tp_%s" % str(i))
    with open(fname, "w") as f:
        for i in range(0, len(y_true)):
            f.write("%s |$| %s |$| %s |$| %s\n" %(y_true[i], y_pred[i], name[i], vid[i]))

    fname = os.path.join(result_dir, "confusion_matrix")
    with open(fname, "a") as f:
        f.write(np.array_str(confusion_matrix(y_true, y_pred, labels=["Positive", "Negative"])) + "\n\n")

def describe_classif_reports(results, result_dir):

    """
    Function to make classification stats report.

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


def log_pred_probability(df_feature_test, y_pred, test_mani, clf, result_dir, tag):

    y_pred_prob = clf.predict_proba(df_feature_test)
    fname = os.path.join(result_dir, "predict_prob_" + str(tag))

    with open(fname, "w") as f:
        class_names = [str(x) for x in clf.classes_]
        s = ' |$| '.join(class_names)
        f.write("Truth |$| Pred |$| " + s + " |$| Name |S| VID" + "\n")
        truth_labels = [str(x) for x in list(test_mani.label)]
        pred_labels = [str(x) for x in list(y_pred)]
        truth_names = [str(x) for x in list(test_mani.name)]
        truth_vids = [str(x) for x in list(test_mani.visit_id)]
        for i in range(0, len(y_pred_prob)):
            preds = [str(x) for x in y_pred_prob[i]]
            preds = ' |$| '.join(preds)
            f.write(truth_labels[i] + " |$| " + pred_labels[i] + " |$| " + preds +  " |$| " + truth_names[i] + " |$| " + truth_vids[i] +"\n")

def log_interpretation(df_feature_test, clf, result_dir, tag, cols):

    preds, bias, contributions = ti.predict(clf, df_feature_test)
    fname = os.path.join(result_dir, "interpretation_" + str(tag))
    with open(fname, "w") as f:
        data_dict = {}
        for i in range(len(df_feature_test)):
            name = test_mani.iloc[i]['name']
            vid = str(test_mani.iloc[i]['visit_id'])
            key = name + "_" + str(vid)
            data_dict[key] = {}
            data_dict[key]['name'] = name
            data_dict[key]['vid'] = vid
            c = list(contributions[i,:,0])
            c = [round(float(x), 2) for x in c]
            fn = list(cols)
            fn = [str(x) for x in fn]
            feature_contribution = list(zip(c, fn))
            data_dict[key]['contributions'] = feature_contribution
        f.write(json.dumps(data_dict, indent=4))


def classify(train, test, result_dir, tag, save_model, pred_probability, interpret):

    train_mani = train.copy()
    test_mani = test.copy()
    clf = RandomForestClassifier(n_estimators=100)
    fields_to_remove = ['visit_id', 'name', 'label']
    df_feature_train = train_mani.drop(fields_to_remove, axis=1)
    df_feature_test = test_mani.drop(fields_to_remove, axis=1)

    columns = df_feature_train.columns
    df_feature_train = df_feature_train.to_numpy()
    train_labels = train_mani.label.to_numpy()
    
    # Perform training
    clf.fit(df_feature_train, train_labels)

    # Obtain feature importances
    feature_importances = pd.DataFrame(clf.feature_importances_, index = columns, columns=['importance']).sort_values('importance', ascending=False)
    report_feature_importance(feature_importances, result_dir)

    # Perform classification and get predictions
    cols = df_feature_test.columns
    df_feature_test = df_feature_test.to_numpy()
    y_pred = clf.predict(df_feature_test)
    
    acc = accuracy_score(test_mani.label, y_pred)
    prec = precision_score(test_mani.label, y_pred, pos_label="Positive")
    rec = recall_score(test_mani.label, y_pred, pos_label="Positive")

    # Write accuracy score
    fname = os.path.join(result_dir, "accuracy")
    with open(fname, "a") as f:
        f.write("\nAccuracy score: " + str(round(acc*100, 3)) + "%" + "\n")
        f.write("Precision score: " + str(round(prec*100, 3)) + "%" + "\n")
        f.write("Recall score: " + str(round(rec*100, 3)) + "%" +  "\n")

    print("Accuracy Score:", acc)

    # Save trained model if save_model is True
    if save_model:
        model_fname = os.path.join(result_dir, "model_" + str(tag) + ".joblib")
        joblib.dump(clf, model_fname)
    if pred_probability:
        log_pred_probability(df_feature_test, y_pred, test_mani, clf, result_dir, tag)
    if interpret:
        log_interpretation(df_feature_test, clf, result_dir, cols)

    return list(test_mani.label), list(y_pred), list(test_mani.name), list(test_mani.visit_id)


def classify_crossval(df_labelled, result_dir, save_model, pred_probability, interpret):

    vid_list = df_labelled['visit_id'].unique()
    num_iter = 10
    num_test_vid = int(len(vid_list)/num_iter)
    used_test_ids = []
    results = []

    print("Total Number of visit IDs:", len(vid_list))
    print("Number of visit IDs to use in a fold:", num_test_vid)

    for i in range(0, num_iter):
        print("Performing fold:", i)
        vid_list_iter = list(set(vid_list) - set(used_test_ids))
        chosen_test_vid = random.sample(vid_list_iter, num_test_vid)
        used_test_ids += chosen_test_vid

        df_train = df_labelled[~df_labelled['visit_id'].isin(chosen_test_vid)]
        df_test = df_labelled[df_labelled['visit_id'].isin(chosen_test_vid)]

        fname = os.path.join(result_dir, "composition")
        train_pos = len(df_train[df_train['label'] == 'Positive'])
        test_pos = len(df_test[df_test['label'] == 'Positive'])

        with open(fname, "a") as f:
            f.write("\nFold " + str(i) + "\n")
            f.write("Train: " + str(train_pos) + " " + get_perc(train_pos, len(df_train)) + "\n")
            f.write("Test: " + str(test_pos) + " " + get_perc(test_pos, len(df_test)) + "\n")
            f.write("\n")

        result = classify(df_train, df_test, result_dir, i, save_model, pred_probability, interpret)
        results.append(result)

    return results

def get_perc(num, den):

    return str(round(num/den * 100, 2)) + "%"


def pipeline(feature_file, label_file, result_dir, save_model, pred_probability, interpret):

    """
    Function to run data labelling pipeline.

    Args:
      config_filename: Path of the YAML file with feature and output file paths
    Returns:
      Nothing, creates a result directory with all the results.

    This functions does the following:

    1. Reads input files.
    2. Creates DataFrame out of labelled data.
    3. Performs training/classification.
    4. Write results to specified output files.
    """

    df_features = pd.read_csv(feature_file)
    df_labels = pd.read_csv(label_file)
    result_dir.mkdir(parents=True, exist_ok=True)

    df_labelled = df_features.merge(df_labels[['visit_id', 'name', 'label']], on=['visit_id', 'name'])
    
    df_positive = df[df['label'] == 'Positive']
    df_negative = df[df['label'] == 'Negative']

    fname = os.path.join(result_dir, "composition")
    with open(fname, "a") as f:
        f.write("Number of samples: " + str(len(df)) + "\n")
        f.write("Labelled samples: " + str(len(df_labelled)) + " " + get_perc(len(df_labelled), len(df)) + "\n")
        f.write("Positive samples: " + str(len(df_positive)) + " " + get_perc(len(df_positive), len(df)) + "\n")
        f.write("Negative samples: " + str(len(df_negative)) + " " + get_perc(len(df_negative), len(df)) + "\n")
        f.write("\n")

    results = classify_crossval(df_labelled, result_dir, folds, save_model, pred_probability, interpret)
    report = describe_classif_reports(results, result_dir)

def main(program: str, args: List[str]):
    
    parser = argparse.ArgumentParser(prog=program, description="Run the WebGraph classification pipeline.")
    
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
    parser.add_argument(
        "--save",
        type=bool,
        help="Save trained model file.",
        default=False
    )
    parser.add_argument(
        "--probability",
        type=bool,
        help="Log prediction probabilities.",
        default=False
    )
    parser.add_argument(
        "--interpret",
        type=bool,
        help="Log results of tree interpreter.",
        default=False
    )

    ns = parser.parse_args(args)
    pipeline(ns.features, ns.labels, ns.out, ns.save, ns.probability, ns.interpret)


if __name__ == "__main__":

    main(sys.argv[0], sys.argv[1:])





