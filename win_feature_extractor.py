import numpy as np
import pandas as pd

from kmodes.kmodes import KModes
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from typing import List, Optional

from data_dragon import DataDragon
from lol_rules import TURRET_TYPES, LANES, TURRET_TIERS, DRAGON_TYPES, CHAMPION_ROLES, WARD_TYPES

POSSIBLE_FEATURES = ['time', 'gold', 'kills', 'deaths',
                     'level', 'level_mean',
                     'respawn', 'alive',
                     'champions_n_hot', 'champions_k_modes'
                     'dragons', 'dragons_total',
                     'barons', 'epic_buffs', 'monsters',
                     'turrets', 'turrets_per_lane', 'turrets_per_tier', 'turrets_total',
                     'inhibitors_per_lane', 'inhibitors_total', 'inhibitors_respawn',
                     'wards_total', 'wards_per_type']


class WinFeatureExtractor:
    champion_data: dict
    features: List[str]
    normalize: bool
    scaler: StandardScaler
    km: Optional[KModes]

    def __init__(self, features: List[str], data_dragon: DataDragon, normalize: bool = False):
        self.champion_data = data_dragon.get_champion_data()
        self.features = features
        self.normalize = normalize
        self.scaler = StandardScaler()
        self.km = None

    def get_features(self, x: pd.DataFrame) -> np.ndarray:
        columns = []  # time

        champion_names = list(sorted(self.champion_data['data'].keys()))

        for feature in self.features:
            if feature == 'time':
                columns.append(x['time'].values / 60000)
            elif feature == 'gold':
                for team in (1, 2):
                    columns.append(x[f't{team}_total_gold'].values / 1000)
            elif feature == 'kills':
                for team in (1, 2):
                    columns.append(x[f't{team}_champion_kills'].values)
            elif feature == 'deaths':
                for team in (1, 2):
                    columns.append(x[f't{team}_champion_deaths'].values)
            elif feature == 'barons':
                for team in (1, 2):
                    columns.append(x[f't{team}_baron_kills'].values)
            elif feature == 'epic_buffs':
                for team in (1, 2):
                    for epic_monster in ('dragon', 'baron'):
                        columns.append(x[f't{team}_{epic_monster}_buff_remaining'].values / 1000)
            elif feature == 'level':
                for team in (1, 2):
                    for role in CHAMPION_ROLES:
                        columns.append(x[f't{team}_{role}_level'].values)
            elif feature == 'level_mean':
                for team in (1, 2):
                    columns.append(x[[f't{team}_{role}_level' for role in CHAMPION_ROLES]].values.mean(axis=1))
            elif feature == 'respawn':
                for team in (1, 2):
                    for role in CHAMPION_ROLES:
                        columns.append(x[f't{team}_{role}_respawn'].values)
            elif feature == 'alive':
                for team in (1, 2):
                    columns.append((x[[f't{team}_{role}_respawn' for role in CHAMPION_ROLES]].values == 0).sum(axis=1))
            elif feature == 'champions_n_hot':
                for champion_name in champion_names:
                    champions_present = np.array([(champion_name == x[[f't{team}_{role}_champion' for role in CHAMPION_ROLES]].values).sum(axis=1)
                                                  for team in (1, 2)], dtype=np.float64)
                    columns.append(champions_present[0] - champions_present[1])
            elif feature == 'champions_k_modes':
                champions = [x[[f't{team}_{role}_champion' for role in CHAMPION_ROLES]].values for team in (1, 2)]
                samples = np.concatenate(champions)
                n_clusters = 4
                if not self.km:
                    # One sample per game is enough to calculate composition clusters
                    one_per_game_x = x[x['time'] == 60_000]
                    one_minute_samples = [one_per_game_x[[f't{team}_{role}_champion' for role in CHAMPION_ROLES]].values for team in (1, 2)]
                    one_minute_samples = np.concatenate(one_minute_samples)
                    self.km = KModes(n_clusters=n_clusters, init='Huang', n_init=5, verbose=1).fit(one_minute_samples)
                clusters = self.km.predict(samples)
                for cluster_idx in range(n_clusters):
                    columns.append((clusters[:x.shape[0]] == cluster_idx).astype(np.float64) - (clusters[x.shape[0]:] == cluster_idx).astype(np.float64))
            elif feature == 'dragons':
                for team in (1, 2):
                    for dragon_type in DRAGON_TYPES:
                        columns.append(x[f't{team}_{dragon_type}_dragon_killed'].values)
            elif feature == 'dragons_total':
                for team in (1, 2):
                    columns.append(x[[f't{team}_{dragon_type}_dragon_killed' for dragon_type in DRAGON_TYPES]].values.sum(axis=1))
            elif feature == 'monsters':
                for team in (1, 2):
                    columns.append(x[[f't{team}_{dragon_type}_dragon_killed' for dragon_type in DRAGON_TYPES] + [f't{team}_baron_kills']].values.sum(axis=1))
            elif feature == 'turrets':
                for team in (1, 2):
                    for turret_type in TURRET_TYPES:
                        columns.append(x[f't{team}_{turret_type}'].values)
            elif feature == 'turrets_per_lane':
                for team in (1, 2):
                    for lane in LANES:
                        columns.append(x[[f't{team}_turret_{turret_tier}_{lane}' for turret_tier in TURRET_TIERS] +
                                         ([f't{team}_turret_nexus_mid'] if lane == 'mid' else [])].values.sum(axis=1))
            elif feature == 'turrets_per_tier':
                for team in (1, 2):
                    for turret_tier in TURRET_TIERS:
                        columns.append(x[[f't{team}_turret_{turret_tier}_{lane}' for lane in LANES]].values.sum(axis=1))
                    columns.append(x[f't{team}_turret_nexus_mid'].values)
            elif feature == 'turrets_total':
                for team in (1, 2):
                    columns.append(x[[f't{team}_{turret_type}' for turret_type in TURRET_TYPES]].values.sum(axis=1))
            elif feature == 'inhibitors_per_lane':
                for team in (1, 2):
                    for lane in LANES:
                        columns.append(x[f't{team}_inhibitor_{lane}'].values)
            elif feature == 'inhibitors_total':
                for team in (1, 2):
                    columns.append(x[[f't{team}_inhibitor_{lane}' for lane in LANES]].values.sum(axis=1))
            elif feature == 'inhibitors_respawn':
                for team in (1, 2):
                    for lane in LANES:
                        columns.append(x[f't{team}_inhibitor_{lane}_respawn'].values / 1000)
            elif feature == 'wards_total':
                for team in (1, 2):
                    columns.append(x[[f't{team}_{ward_type}_wards' for ward_type in WARD_TYPES]].values.sum(axis=1))
            elif feature == 'wards_per_type':
                for team in (1, 2):
                    for ward_type in WARD_TYPES:
                        columns.append(x[f't{team}_{ward_type}_wards'].values)
            else:
                raise NotImplementedError
        features = np.stack(columns, axis=1)
        if self.normalize:
            try:
                features = self.scaler.transform(features)
            except NotFittedError as e:
                self.scaler.fit(features)
                features = self.scaler.transform(features)
        return features
