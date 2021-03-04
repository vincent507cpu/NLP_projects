import pandas as pd
import os
from sklearn.model_selection import train_test_split

BASE_DIR = '/home/nlp/Documents/dataset/qa'


def train_val_split(train_x_seg_path, train_y_seg_path, val_x_seg_path, val_y_seg_path):
    train_df_x = pd.read_csv(train_x_seg_path, encoding='utf-8')
    train_df_y = pd.read_csv(train_y_seg_path, encoding='utf-8')
    print(len(train_df_x))
    print(len(train_df_y))
    # split train, validation
    X_train, X_val, y_train, y_val = train_test_split(train_df_x, train_df_y,
                                                      test_size=0.002,  # 8W*0.002
                                                      )

    X_train.to_csv(train_x_seg_path, sep='\t', index=None, header=False)
    y_train.to_csv(train_y_seg_path, sep='\t', index=None, header=False)
    X_val.to_csv(val_x_seg_path, sep='\t', index=None, header=False)
    y_val.to_csv(val_y_seg_path, sep='\t', index=None, header=False)


if __name__ == "__main__":
    train_val_split('{}/train_set.seg_x.txt'.format(BASE_DIR),
                    '{}/train_set.seg_y.txt'.format(BASE_DIR),
                    '{}/val_set.seg_x.txt'.format(BASE_DIR),
                    '{}/val_set.seg_y.txt'.format(BASE_DIR))