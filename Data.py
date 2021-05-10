from sklearn import datasets
import pandas as pd
import numpy as np


def example_chapter6():
    df = pd.DataFrame([
                        [42.0, 'male', 12.0, 'reading', 'class2'],
                        [35.0, 'U', 3.0, 'cooking', 'class1'],
                        [0.0, 'female', 7.0, 'cycling', 'class3'],
                        [0.0, 'U', 0.0, 'U', 'class4']
                    ])
    df.columns = ['age', 'gender', ' month_birth', 'hobby', 'target']

    return df

    
def boston():
    raw_boston = datasets.load_boston()
    X_boston = pd.DataFrame(raw_boston.data)
    y_boston = pd.DataFrame(raw_boston.target)
    df_boston = pd.concat([X_boston, y_boston], axis=1)

    col_boston = np.append(raw_boston.feature_names, ['target'])
    df_boston.columns = col_boston

    return df_boston