from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from .utils import get_domain


tqdm.pandas()


def get_final_label(row):
    cp_label = row['cat_id_cookiepedia']
    tranco_label = row['cat_id_tranco']

    if pd.isna(cp_label) and pd.isna(tranco_label):
        return pd.NA

    if pd.isna(cp_label):
        return tranco_label

    if pd.isna(tranco_label):
        return cp_label

    return cp_label if cp_label > tranco_label else tranco_label


def clean_up_final_label(label):
    if label < 0 or label > 3:
        return pd.NA

    return label


def get_categories(cookiepedia_path: Path, tranco_path: Path):
    df_cookiepedia = pd.read_csv(str(cookiepedia_path), index_col=0)
    df_tranco = pd.read_csv(str(tranco_path), index_col=0)

    df_cookiepedia = df_cookiepedia[['name', 'domain', 'cat_id']]
    df_tranco = df_tranco[['name', 'domain', 'cat_id']]

    df_cookiepedia.drop_duplicates(inplace=True)
    df_tranco.drop_duplicates(inplace=True)

    return df_cookiepedia, df_tranco


def process_declared_labels(cookiepedia_path: Path, tranco_path: Path):
    df_cookiepedia, df_tranco = get_categories(cookiepedia_path, tranco_path)
    df_labels = pd.merge(df_tranco, df_cookiepedia, how = 'outer', on=['name', 'domain'], suffixes=('_cookiepedia', '_tranco'))
    df_labels['declared_label'] = df_labels.progress_apply(get_final_label, axis=1)
    df_labels.drop_duplicates(inplace=True)
    df_labels = df_labels['declared_label'].groupby([df_labels.name, df_labels.domain]).apply(list).reset_index()
    df_labels['declared_label'] = df_labels.progress_apply(lambda x: max(x['declared_label']), axis=1)
    df_labels['domain'] = df_labels.domain.progress_apply(get_domain)
    df_labels['declared_label'] = df_labels.declared_label.progress_apply(clean_up_final_label)

    return df_labels


def get_cookiepedia_categories():
    URL = "https://raw.githubusercontent.com/shaoormunir/cookiewatcher-groundtruth/main/cookies_categories_updated.csv?token=AAGTHXNTOH6LH4P56CLLPTTB22WRG"
    df_cookiepedia = pd.read_table(URL, sep=",")
    return df_cookiepedia


def label_cookiepedia_data(category):
    return category == 'Targeting/Advertising'


def label_storage_cookiepedia(df, cookiepedia_path: Path, tranco_path: Path):
    df_storage = df[df['type'] == 'Storage']
    df_cookiepedia = get_categories(cookiepedia_path, tranco_path)
    df_merged = df_storage.merge(df_cookiepedia, left_on='name', right_on='cookie_key')
    df_merged = df_merged[['visit_id', 'name', 'category']]
    df_merged = df_merged.drop_duplicates()
    df_merged['cookiepedia_label'] = df_merged['category'].apply(label_cookiepedia_data)

    return df_merged


def label_storage_declared(df, df_labels):
    df_storage = df[(df['type'] == 'Storage') & (df['attr'] == 'Cookie')]
    df_storage = df_storage[df_storage['domain'].notnull()]
    df_merged = pd.merge(df_storage, df_labels, how='left', on=['name', 'domain'])
    df_merged = df_merged[['visit_id', 'name', 'declared_label']]

    return df_merged


