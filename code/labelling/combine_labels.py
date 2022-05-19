import pandas as pd


def get_combined_label(row):
    setter_label = row['setter_label']
    declared_label = row['declared_label']

    if declared_label == 3:
        return 'Positive'
    if setter_label is False:
        return 'Negative'
    return 'Unknown'


def label_storage_nodes(df_setters, df_declared):
    df_combined_data = pd.merge(df_setters, df_declared, on=['visit_id', 'name'], how='outer')
    df_combined_data['label'] = df_combined_data.apply(get_combined_label, axis=1)
    df_combined_data = df_combined_data.drop_duplicates()

    return df_combined_data
