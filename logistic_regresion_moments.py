import cv2
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

from data_dragon import DataDragon
from lol_rules import DRAGON_TYPES, TURRET_TYPES
from win_feature_extractor import WinFeatureExtractor
from logistic_regression_win_predictor import LogisticRegressionWinPredictor


def get_max_deviations(probabilities,number_of_moments):
    
    differences = np.abs(probabilities[1:, :] - probabilities[:-1, :])

    diff_abs = np.abs(differences[:, 0] - differences[:, 1])
    indices_of_largest_diffs = np.argpartition(diff_abs, -number_of_moments)[-number_of_moments:]

    indices_sorted_by_diffs = indices_of_largest_diffs[np.argsort(diff_abs[indices_of_largest_diffs])]
    indices_sorted = np.sort(indices_of_largest_diffs)
    print(indices_sorted)
    return indices_sorted

def get_log_reg(game_id,fow_model=True):
    
    data_folder = os.path.join(os.path.dirname(__file__),'Game_data')
    data_dragon = DataDragon(data_folder, '13.4')
    dataset_folder = os.path.join(os.path.dirname(__file__), 'data_fow' if fow_model else 'data_muller')

    
    output_folder = os.path.join('output', 'win_prediction')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    output_folder = os.path.join(output_folder, 'win_visualization')
    df = pd.read_csv(os.path.join(dataset_folder, 'win_dataset.csv'))
    game_ids = np.unique(df['id'])
    kf = KFold(n_splits=5)

    model = train_model(game_ids,df, data_dragon, kf)
    
    id = game_ids[np.where(game_ids == game_id)[0][0]]
    x_test = df.loc[df['id'] == id]
    y_test = x_test.pop('winner').values
    x_test.pop('id')
    probabilities = model.predict_proba(x_test)
    return probabilities


def train_model(game_ids,df,data_dragon,kf):

    features = ['kills', 'turrets_total', 'monsters', 'gold']
    for train_index, test_index in kf.split(game_ids):
        # Prepare train data
        rows = []
        for id in game_ids[train_index]:
            game_data = df.loc[df['id'] == id]
            times = [timestep for timestep in game_data['time']]
            for timestep in times:
                rows.extend(game_data.loc[game_data['time'] == timestep].index)
        x_train = df.iloc[rows]
        y_train = x_train.pop('winner').values
        x_train.pop('id')
        feature_extractor = WinFeatureExtractor(features,
                                                data_dragon,
                                                normalize=False)
        model = LogisticRegressionWinPredictor(feature_extractor)
    model.train(x_train, y_train)
    return model