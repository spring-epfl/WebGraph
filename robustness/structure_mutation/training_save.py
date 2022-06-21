from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn import preprocessing
import pandas as pd
import os
import sys
from yaml import load, dump
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from treeinterpreter import treeinterpreter as ti
import json
from collections import Counter
import random
import joblib

def print_stats(report, result_dir, avg='macro avg', stats=['mean', 'std']):

  by_label = report.groupby('label').describe()
  fname = os.path.join(result_dir, "scores")
  with open(fname, "w") as f:
    for stat in stats:
      print(by_label.loc[avg].xs(stat, level=1))
      x = by_label.loc[avg].xs(stat, level=1)
      f.write(by_label.loc[avg].xs(stat, level=1).to_string())
      f.write("\n")
    
def report_feature_importance(feature_importances, result_dir):
  
  fname = os.path.join(result_dir, "featimp")
  with open(fname, "a") as f:
      f.write(feature_importances.to_string())
      f.write("\n")
    
def report_true_pred(y_true, y_pred, name, vid, i, result_dir):
  
  fname = os.path.join(result_dir, "tp_%s" % str(i))
  with open(fname, "w") as f:
    for i in range(0, len(y_true)):
      f.write("%s |$| %s |$| %s |$| %s\n" %(y_true[i], y_pred[i], name[i], vid[i]))

def describe_classif_reports(results, result_dir):
    
  true_vectors, pred_vectors, name_vectors, vid_vectors = [r[0] for r in results], [r[1] for r in results], [r[2] for r in results], [r[3] for r in results]

  all_folds = pd.DataFrame(columns=['label', 'fold', 'precision', 'recall', 'f1-score', 'support'])
  for i, (y_true, y_pred, name, vid) in enumerate(zip(true_vectors, pred_vectors, name_vectors, vid_vectors)):
    report_true_pred(y_true, y_pred, name, vid, i, result_dir)
    output = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(output).transpose().reset_index().rename(columns={'index': 'label'})
    df['fold'] = i
    all_folds = all_folds.append(df)
  return all_folds

def classify_robust(chosen_test_vid, vid_list, df_labelled, df_all_test, feature_config, result_dir, output_prob=False, save_model=False):

  num_iter = 1
  results = []
  feature_type_set = feature_config['feature_set']
  feature_list = []
  used_test_ids = []
  for feature_type in feature_type_set:
    feature_list += feature_config[feature_type]
  fname = os.path.join(result_dir, "used_features")
  with open(fname, 'w') as f:
    for feature in feature_list:
      f.write(feature + "\n")
  for i in range(0, num_iter):
    print("Fold", i)
    if save_model:
      test_vid_fname = os.path.join(result_dir, "test_id_" + str(i))
    train_vid_list = list(set(vid_list) - set(chosen_test_vid))
    df_train = df_labelled[df_labelled['visit_id'].isin(train_vid_list)]
    df_test = df_all_test[df_all_test['visit_id'].isin(chosen_test_vid)]
    #print(df_train.shape)
    #print(df_test.shape)
    result = classify_fs(df_train, df_test, feature_list, result_dir, i, output_prob)
    results.append(result)
  return results

def classify_diff(vid_list, df_labelled, df_all_test, feature_config, result_dir, output_prob=False, save_model=False):

  num_iter = 10
  original_vid_list = list(range(0, 10000))
  vid_list = list(set(original_vid_list) - set(bad_vid_list))
  num_test_vid = int(len(vid_list)/num_iter)
  results = []
  feature_type_set = feature_config['feature_set']
  feature_list = []
  used_test_ids = []
  for feature_type in feature_type_set:
    feature_list += feature_config[feature_type]
  fname = os.path.join(result_dir, "used_features")
  with open(fname, 'w') as f:
    for feature in feature_list:
      f.write(feature + "\n")
  for i in range(0, num_iter):
    #print("Fold", i)
    vid_list_iter = list(set(vid_list) - set(used_test_ids))
    chosen_test_vid = random.sample(vid_list_iter, num_test_vid)
    used_test_ids += chosen_test_vid
    #print(len(used_test_ids))
    if save_model:
      test_vid_fname = os.path.join(result_dir, "test_id_" + str(i))
      with open(test_vid_fname, 'w') as f:
        data_dict = {'test_vids' : chosen_test_vid}
        f.write(json.dumps(data_dict))
    df_train = df_labelled[~df_labelled['visit_id'].isin(chosen_test_vid)]
    df_test = df_all_test[df_all_test['visit_id'].isin(chosen_test_vid)]
    #print(df_train.shape)
    #print(df_test.shape)
    df_train = df_train.sample(n=900000)
    df_test = df_test.sample(n=85000)
    #print(df_train.shape)
    #print(df_test.shape)
    result = classify_fs(df_train, df_test, feature_list, result_dir, i, output_prob)
    results.append(result)
  return results


def classify_with_model(vid_list, df_all_test, feature_config, result_dir, trained_model_dir):

  results = []
  feature_type_set = feature_config['feature_set']
  feature_list = []
  for feature_type in feature_type_set:
    feature_list += feature_config[feature_type]
  fname = os.path.join(result_dir, "used_features")
  with open(fname, 'w') as f:
    for feature in feature_list:
      f.write(feature + "\n")
  num_iter = 10
  for i in range(0, num_iter):
    print("Fold", i)
    model_fname = os.path.join(trained_model_dir, "model_" + str(i) + ".joblib")
    test_vid_fname = os.path.join(trained_model_dir, "test_id_" + str(i))
    with open(test_vid_fname) as f:
      data_dict = json.loads(f.read())
      test_vids = data_dict['test_vids']
    df_test = df_all_test[df_all_test['visit_id'].isin(test_vids)]
    clf = joblib.load(model_fname)
    result = classify_clf(clf, df_test, feature_list, result_dir)
    results.append(result)
  return results



def classify_clf(clf, test, feature_list, result_dir):

  cpt = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0]
  test_mani = test.copy()
  feature_list_mani = feature_list.copy()
  #fields_to_remove = ['visit_id', 'name', 'top_level_url', 'url_hostname', 'url_ps1', 'top_level_ps1', 'label']
  fields_to_remove = ['visit_id', 'name', 'top_level_url', 'label']

  if 'content_policy_type' in feature_list:
    test_mani['content_policy_type'] = pd.Categorical(test_mani['content_policy_type'], categories=cpt)
    test_mani = pd.get_dummies(test_mani, prefix=['content_policy_type'], columns=['content_policy_type'])
  
  df_feature_test = test_mani.drop(fields_to_remove, axis=1)

  if 'content_policy_type' in feature_list:
    cp_cols = [col for col in df_feature_test.columns if 'content_policy_type' in col]
    feature_list_mani.remove('content_policy_type')
    feature_list_mani += cp_cols

  df_feature_test = df_feature_test[feature_list_mani]
  y_pred = clf.predict(df_feature_test)
  acc = accuracy_score(test_mani.label, y_pred)

  fname = os.path.join(result_dir, "accuracy")
  with open(fname, "a") as f:
      f.write(str(acc))
      f.write("\n")
  print("Accuracy Score:", acc)

  return list(test_mani.label), list(y_pred), list(test_mani.name), list(test_mani.visit_id)

def classify_fs(train, test, feature_list, result_dir, tag, output_prob=False, save_model=True):

  cpt = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0]
  test_copy = test.copy()

  train_mani = train.copy()
  test_mani = test.copy()
  feature_list_mani = feature_list.copy()

  #print(len(feature_list_mani))

  if 'content_policy_type' in feature_list:
    train_mani['content_policy_type'] = pd.Categorical(train_mani['content_policy_type'], categories=cpt)
    train_mani = pd.get_dummies(train_mani, prefix=['content_policy_type'], columns=['content_policy_type'])
    test_mani['content_policy_type'] = pd.Categorical(test_mani['content_policy_type'], categories=cpt)
    test_mani = pd.get_dummies(test_mani, prefix=['content_policy_type'], columns=['content_policy_type'])

  #print(train_mani.shape)
  #print(test_mani.shape)

  clf = RandomForestClassifier(n_estimators=100)
  fields_to_remove = ['visit_id', 'name', 'top_level_url', 'url_hostname', 'url_ps1', 'top_level_ps1', 'label']
  df_feature_train = train_mani.drop(fields_to_remove, axis=1)

  #print(df_feature_train.shape)
    
  if 'content_policy_type' in feature_list:
    cp_cols = [col for col in df_feature_train.columns if 'content_policy_type' in col]
    feature_list_mani.remove('content_policy_type')
    feature_list_mani += cp_cols
  df_feature_train = df_feature_train[feature_list_mani]

  #print(df_feature_train.shape)

  clf.fit(df_feature_train, train_mani.label)


  if save_model:
    model_fname = os.path.join(result_dir, "model_" + str(tag) + ".joblib")
    joblib.dump(clf, model_fname)
  
  feature_importances = pd.DataFrame(clf.feature_importances_, index = df_feature_train.columns, columns=['importance']).sort_values('importance', ascending=False)
  report_feature_importance(feature_importances, result_dir)
  df_feature_test = test_mani.drop(fields_to_remove, axis=1)

  # if 'content_policy_type' in feature_list:
  #   cp_cols = [col for col in df_feature_test.columns if 'content_policy_type' in col]
  #   #feature_list_mani.remove('content_policy_type')
  #   feature_list += cp_cols
  #print(df_feature_test.shape)
  df_feature_test = df_feature_test[feature_list_mani]
  #print(df_feature_test.shape)
  y_pred = clf.predict(df_feature_test)
  acc = accuracy_score(test_mani.label, y_pred)

  fname = os.path.join(result_dir, "accuracy")
  with open(fname, "a") as f:
      f.write(str(acc))
      f.write("\n")
  print("Accuracy Score:", acc)

  if output_prob:
    y_pred_prob = clf.predict_proba(df_feature_test)
    fname = os.path.join(result_dir, "predict_prob_" + str(tag))
    with open(fname, "w") as f:
      class_names = [str(x) for x in clf.classes_]
      s = ' |$| '.join(class_names)
      f.write("Truth |$| Pred |$| " + s + " |$| Name |S| VID |$| Content_Type" + "\n")
      truth_labels = [str(x) for x in list(test_mani.label)]
      pred_labels = [str(x) for x in list(y_pred)]
      truth_names = [str(x) for x in list(test_mani.name)]
      truth_vids = [str(x) for x in list(test_mani.visit_id)]
      truth_content = [str(x) for x in list(test_copy.content_policy_type)]
      for i in range(0, len(y_pred_prob)):
          preds = [str(x) for x in y_pred_prob[i]]
          preds = ' |$| '.join(preds)
          f.write(truth_labels[i] + " |$| " + pred_labels[i] + " |$| " + preds +  " |$| " + truth_names[i] + " |$| " + truth_vids[i] + " |$| " + truth_content[i] +"\n")

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
        fn = list(df_feature_test.columns)
        fn = [str(x) for x in fn]
        feature_contribution = list(zip(c, fn))
        #feature_contribution = list(zip(contributions[i,:,0], df_feature_test.columns))
        data_dict[key]['contributions'] = feature_contribution
      f.write(json.dumps(data_dict, indent=4))

        # f.write('instance:' + test.iloc[i]['name'] + "\n")
        # for c, feature in sorted(zip(contributions[i,:,0], df_feature_test.columns)):
        #   f.write(feature + " " + str(round(c, 2)) + "\n")

  return list(test_mani.label), list(y_pred), list(test_mani.name), list(test_mani.visit_id)

if __name__ == "__main__":

  with open('bad_ids.json') as f:
      data_dict = json.loads(f.read())
  bad_vid_list = data_dict['bad_id_all']
  #print(len(bad_vid_list))

  filepath_file = "filepaths.yaml"
  with open(filepath_file) as f:
    filepath_config = load(f.read())

  feature_file = filepath_config['feature_config']
  with open(feature_file) as f:
      feature_config = load(f.read())

  training_type = sys.argv[1]
  LABELLED_DIR = filepath_config['labelled_dir']

  fnames = os.listdir(LABELLED_DIR)
  df_labelled = pd.DataFrame()
  for fname in fnames:  
    df_labelled_file = pd.read_csv(os.path.join(LABELLED_DIR, fname))
    df_labelled = df_labelled.append(df_labelled_file)

  if training_type == "save_model":
    RESULT_DIR = filepath_config['result_traintest']
    if not os.path.exists(RESULT_DIR):
      os.mkdir(RESULT_DIR)
    original_vid_list = list(range(0, 10000))
    vid_list = list(set(original_vid_list) - set(bad_vid_list))
    results = classify_diff(vid_list, df_labelled, df_labelled, feature_config, RESULT_DIR, output_prob=False)
    report = describe_classif_reports(results, RESULT_DIR)
    print_stats(report, RESULT_DIR)

  if training_type == "use_model":
    RESULT_DIR = filepath_config['result_traintest']
    TRAINED_MODEL_DIR = filepath_config['trained_model']
    if not os.path.exists(RESULT_DIR):
      os.mkdir(RESULT_DIR)
    TEST_DIR = filepath_config['test_dir']
    fnames = os.listdir(TEST_DIR)
    df_all_test = pd.DataFrame()
    for fname in fnames:  
      df_test_file = pd.read_csv(os.path.join(TEST_DIR, fname))
      df_all_test = df_all_test.append(df_test_file)

    original_vid_list = list(range(0, 10000))
    vid_list = list(set(original_vid_list) - set(bad_vid_list))
    results = classify_with_model(vid_list, df_all_test, feature_config, RESULT_DIR, TRAINED_MODEL_DIR)
    report = describe_classif_reports(results, RESULT_DIR)
    print_stats(report, RESULT_DIR)

  if training_type == "save_model_robust":
    RESULT_DIR = filepath_config['result_traintest']
    if not os.path.exists(RESULT_DIR):
      os.mkdir(RESULT_DIR)
    original_vid_list = list(range(0, 10000))
    vid_list = list(set(original_vid_list) - set(bad_vid_list))
    with open('/home/ubuntu/rob300/chosen_vids.json') as f:
      test_vid_list = json.loads(f.read())['vids']
    results = classify_robust(test_vid_list, vid_list, df_labelled, df_labelled, feature_config, RESULT_DIR, output_prob=False)
    report = describe_classif_reports(results, RESULT_DIR)
    print_stats(report, RESULT_DIR)

  else:
    print("Invalid training type")
  
