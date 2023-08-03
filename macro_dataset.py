import json
import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from tqdm import tqdm
import os

class MacroDataset(Dataset):
    history_size: int
    games: List[dict]
    game_lengths: List[int]
    game_name: List[str]

    def __init__(self, filenames: List[str], history_size: int):
        self.history_size = history_size
        self.games = []
        self.game_lengths = []
        self.game_name =[]
        for i, filename in enumerate(tqdm(filenames)):
            with open(filename, 'r') as f:
                samples = [json.loads(line) for line in f]
                
                
            # Delete the record at index 6
            del samples[6]
            # Initialize an empty dictionary to store our stacked samples
            stacked_samples = {}

            # Fetch the keys from the first sample as our reference for all samples
            sample_keys = samples[0].keys()

            # Iterate over each key
            for key in sample_keys:
                # Prepare a list to collect all values corresponding to the current key
                # across all samples
                sample_values = []

                # Go through all samples
                for sample in samples:
                    # Append the value for the current key from the current sample
                    sample_value = sample[key]
                    sample_values.append(sample_value)

                # # Check the length of sample_values
                # print(f"Key: {key}, Length of sample_values: {len(sample_values)}")

                # Check if the key contains the substring 'types' or if it's 'target_position'
                if 'types' in key or key == 'target_position':
                    # If yes, convert the list of values into a tensor of long integers
                    tensor_values = torch.LongTensor(sample_values)
                else:
                    # If not, convert the list of values into a tensor of floating point numbers
                    tensor_values = torch.FloatTensor(sample_values)

                # Add the created tensor to our result dictionary
                stacked_samples[key] = tensor_values
                
            self.games.append(stacked_samples)
            self.game_lengths.append(len(samples) - (history_size - 1))
            file_name = os.path.basename(filename) 

            # use os.path again to get the file name without the extension
            base_name = os.path.splitext(file_name)[0] 
            self.game_name.append(int(base_name))

    def __len__(self):
        return sum(self.game_lengths) * 10  # 10 targets per item

    def __getitem__(self, item: int):
        target_idx = item % 10
        remaining = item // 10
        for i, game_length in enumerate(self.game_lengths):
            if remaining - game_length < 0:
                sample = {key: self.games[i][key][remaining: remaining + self.history_size]
                          for key in self.games[i].keys()}
                self.add_ally_feature(sample, target_idx)
                return sample, target_idx
            remaining -= game_length

    def get_feature_mean_std(self) -> Tuple[dict, dict]:
        # Calculate the mean and std of each feature for normalization
        feature_means = {key: torch.zeros(value.shape[-1] if key != 'map' else value.shape[-3])
                         for key, value in self.games[0].items()
                         if not ('types' in key or 'target' in key)}
        feature_counts = {key: 0 for key in feature_means.keys()}
        for game in self.games:
            for key in feature_means.keys():
                game_data = game[key]
                if key == 'map':
                    game_data = game_data.swapaxes(-3, -1)
                all_items = game_data.reshape(-1, game_data.shape[-1])
                feature_means[key] += all_items.sum(0)
                feature_counts[key] += all_items.shape[0]
        feature_means = {key: value / feature_counts[key] for key, value in feature_means.items()}
        squared_error_sums = {key: torch.zeros_like(value) for key, value in feature_means.items()}
        for game in self.games:
            for key in feature_means.keys():
                game_data = game[key]
                if key == 'map':
                    game_data = game_data.swapaxes(-3, -1)
                all_items = game_data.reshape(-1, game_data.shape[-1])
                squared_error_sums[key] += (all_items - feature_means[key]).pow(2).sum(0)
        feature_stds = {key: torch.sqrt(value / feature_counts[key]) for key, value in squared_error_sums.items()}
        return feature_means, feature_stds

    def get_target_mean_std(self) -> Tuple[dict, dict]:
        # Calculate the mean and std of each target
        target_means = {key: 0 for key, value in self.games[0].items()
                         if 'target' in key and key != 'target_position'}
        target_counts = {key: 0 for key in target_means.keys()}
        for game in self.games:
            for key in target_means.keys():
                game_data = game[key]
                target_means[key] += game_data.sum()
                target_counts[key] += game_data.numel()
        target_means = {key: value / target_counts[key] for key, value in target_means.items()}
        squared_error_sums = {key: torch.zeros_like(value) for key, value in target_means.items()}
        for game in self.games:
            for key in target_means.keys():
                game_data = game[key]
                squared_error_sums[key] += (game_data - target_means[key]).pow(2).sum()
        target_stds = {key: torch.sqrt(value / target_counts[key]) for key, value in squared_error_sums.items()}
        return target_means, target_stds

    def add_ally_feature(self, sample: dict, champion_idx):
        if champion_idx >= 5:
            champion_in_team_feature = torch.arange(10) >= 5
        else:
            champion_in_team_feature = torch.arange(10) < 5
        mean, var = 0.5, 0.25
        champion_in_team_feature = (champion_in_team_feature.float() - mean) / var
        champion_in_team_feature = champion_in_team_feature.unsqueeze(0).unsqueeze(-1).expand(self.history_size, 10, 1)
        sample['champion_state'] = torch.concat((sample['champion_state'], champion_in_team_feature), dim=-1)

    def normalize(self, mean: dict, std: dict) -> None:
        for game in self.games:
            for key in mean.keys():
                if key == 'map':
                    game[key] = ((game[key].swapaxes(-3, -1) - mean[key]) / std[key]).swapaxes(-3, -1)
                else:
                    game[key] = (game[key] - mean[key]) / std[key]
