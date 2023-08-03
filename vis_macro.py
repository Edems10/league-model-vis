import copy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.express as px
import sys
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import List
from sklearn.model_selection import KFold

from logistic_regression_win_predictor import LogisticRegressionWinPredictor
from win_feature_extractor import WinFeatureExtractor
from data_dragon import DataDragon
from lol_rules import EXTRA_SUMMONER_SPELL_NAMES, MAP_SIZE, WARD_TYPES, DRAGON_TYPES, TURRET_TYPES
from macro_dataset import MacroDataset
from macro_predictor import MacroPredictor
from logistic_regresion_moments import get_log_reg,get_max_deviations


def create_model(device: torch.device,
                 features: List[str],
                 targets: dict,
                 feature_sizes: dict,
                 target_size: int,
                 embedding_sizes: dict,
                 hidden_sizes: dict,
                 history_size: int,
                 model_name: str) -> torch.nn.Module:
    macro_predictor = MacroPredictor(features,
                                     targets,
                                     feature_sizes,
                                     target_size,
                                     embedding_sizes,
                                     hidden_sizes,
                                     history_size)
    macro_predictor.to(device)

    num_params = sum(p.numel() for p in macro_predictor.parameters() if p.requires_grad)

    # Describe the model
    #print(f"Model {model_name} has {num_params} trainable parameters.")
    #print(macro_predictor)

    return macro_predictor


def ablate_features(train_dataset: MacroDataset,
                    valid_dataset: MacroDataset,
                    test_dataset: MacroDataset,
                    batch_size: int,
                    device: torch.device,
                    targets: dict,
                    features: List[str],
                    features_to_ablate: List[str],
                    feature_sizes: dict,
                    embedding_sizes: dict,
                    hidden_sizes: dict,
                    history_size: int,
                    target_size: int,
                    output_folder: str,
                    ) -> None:
    models_to_try = [('All features', features)]
    for feature in features_to_ablate:
        if feature in features:
            models_to_try.append((f'All features - {feature}', [f for f in features if f != feature]))
        else:
            models_to_try.append((f'All features + {feature}', features + [feature]))
    for model_name, sub_features in models_to_try:
        macro_predictor = create_model(device,
                                       sub_features,
                                       targets,
                                       feature_sizes,
                                       target_size,
                                       embedding_sizes,
                                       hidden_sizes,
                                       history_size,
                                       model_name)

        train(macro_predictor, train_dataset, valid_dataset, test_dataset, batch_size, device, targets, output_folder, model_name)


def ablate_targets(train_dataset: MacroDataset,
                   valid_dataset: MacroDataset,
                   test_dataset: MacroDataset,
                   batch_size: int,
                   device: torch.device,
                   targets_dict: dict,
                   targets_to_ablate: List[str],
                   features: List[str],
                   feature_sizes: dict,
                   embedding_sizes: dict,
                   hidden_sizes: dict,
                   history_size: int,
                   output_folder: str
                   ) -> None:
    models_to_try = [('All targets', targets_dict)]
    for target in targets_to_ablate:
        if target not in targets_dict.keys():
            raise ValueError(f"Target to ablate {target} must be in the targets")
        models_to_try.append((f'All targets - {target}', {key: value for key, value in targets_dict.items() if key != target}))
    for model_name, sub_targets_dict in models_to_try:
        target_size = sum(target['size'] for target in sub_targets_dict.values())
        macro_predictor = create_model(device,
                                       features,
                                       sub_targets_dict,
                                       feature_sizes,
                                       target_size,
                                       embedding_sizes,
                                       hidden_sizes,
                                       history_size,
                                       model_name)

        train(macro_predictor, train_dataset, valid_dataset, test_dataset, batch_size, device, sub_targets_dict, output_folder, model_name)


def train(model: torch.nn.Module,
          train_dataset: MacroDataset,
          valid_dataset: MacroDataset,
          test_dataset: MacroDataset,
          batch_size: int,
          device: torch.device,
          targets: dict,
          output_folder: str,
          model_name: str):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    writer = SummaryWriter(os.path.join(output_folder, 'runs', model_name))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_valid_loss = torch.inf
    best_epoch = -1
    model_path = os.path.join(output_folder, 'temp.pt')

    for epoch in range(1, 6):
        train_accuracies = {'position': 0}
        valid_accuracies = {'position': 0}
        train_losses = {target: 0 for target in targets.keys()}
        valid_losses = copy.deepcopy(train_losses)
        model.train()
        for x, target_idx in tqdm(train_dataloader):
            batch_size = len(target_idx)
            # Move to device
            for key, value in x.items():
                x[key] = value.to(device)

            pred = model(x, target_idx)
            losses = {}
            target_start = 0
            for target_name, target in targets.items():
                # Get batch targets for target champions for the last timestep
                y = x[f'target_{target_name}'][torch.arange(batch_size), -1, target_idx]
                target_prediction = pred[:, target_start: target_start + target['size']].squeeze()
                losses[target_name] = target['loss'](target_prediction, y) * target['weight']

                train_losses[target_name] += losses[target_name].item() * batch_size  # Multiply by # of samples in the batch
                if target_name in train_accuracies:
                    train_accuracies[target_name] += (target_prediction.argmax(1) == y).sum()

                target_start += target['size']
            loss = sum(losses.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for x, target_idx in valid_dataloader:
                batch_size = len(target_idx)
                # Move to device
                for key, value in x.items():
                    x[key] = value.to(device)

                pred = model(x, target_idx)
                losses = {}
                target_start = 0
                for target_name, target in targets.items():
                    # Get batch targets for target champions for the last timestep
                    y = x[f'target_{target_name}'][torch.arange(batch_size), -1, target_idx]
                    target_prediction = pred[:, target_start: target_start + target['size']].squeeze()
                    losses[target_name] = target['loss'](target_prediction, y) * target['weight']

                    valid_losses[target_name] += losses[target_name].item() * batch_size  # Multiply by # of samples in the batch
                    if target_name in valid_accuracies:
                        valid_accuracies[target_name] += (target_prediction.argmax(1) == y).sum()

                    target_start += target['size']

        # Average over samples
        train_losses = {key: value / len(train_dataset) for key, value in train_losses.items()}
        valid_losses = {key: value / len(valid_dataset) for key, value in valid_losses.items()}

        # Divide successes by number of samples
        train_accuracies = {key: value / len(train_dataset) for key, value in train_accuracies.items()}
        valid_accuracies = {key: value / len(valid_dataset) for key, value in valid_accuracies.items()}

        for key in train_losses.keys():
            writer.add_scalar(f'Loss/train/{key}', train_losses[key], epoch)
            writer.add_scalar(f'Loss/valid/{key}', valid_losses[key], epoch)

        for key in train_accuracies.keys():
            writer.add_scalar(f'Accuracy/train/{key}', train_accuracies[key], epoch)
            writer.add_scalar(f'Accuracy/valid/{key}', valid_accuracies[key], epoch)

        total_train_loss = sum(train_losses.values())
        total_valid_loss = sum(valid_losses.values())
        writer.add_scalar('Loss/train/total', total_train_loss, epoch)
        writer.add_scalar('Loss/valid/total', total_valid_loss, epoch)

        # Print info to stdout
        print(f'Epoch {epoch}:')
        train_losses = {key: round(value, 3) for key, value in train_losses.items()}
        valid_losses = {key: round(value, 3) for key, value in valid_losses.items()}
        print(f'Train loss: {total_train_loss:.3f} {train_losses}')
        print(f'Valid loss: {total_valid_loss:.3f} {valid_losses}')

        for name, embedding in (('champion', model.champion_embedding),
                                ('item', model.item_embedding),
                                ('summoner_spell', model.summoner_spell_embedding),
                                ('skill', model.skill_embedding)):
            if embedding:
                vector_norm = embedding.weight.norm(dim=1)
                writer.add_scalar(f'Embedding/{name}/min', vector_norm.min().item(), epoch)
                writer.add_scalar(f'Embedding/{name}/mean', vector_norm.mean().item(), epoch)
                writer.add_scalar(f'Embedding/{name}/max', vector_norm.max().item(), epoch)


        # Save best model
        if total_valid_loss < best_valid_loss:
            torch.save(model.state_dict(), model_path)
            best_valid_loss = total_valid_loss
            best_epoch = epoch

    # Load the best model
    model.load_state_dict(torch.load(model_path))
    model.eval()
    test_losses = {target: 0 for target in targets.keys()}
    test_accuracies = {'position': 0} if 'position' in targets else {}
    # Evaluate best model on test dataset
    with torch.no_grad():
        for x, target_idx in test_dataloader:
            batch_size = len(target_idx)
            # Move to device
            for key, value in x.items():
                x[key] = value.to(device)

            pred = model(x, target_idx)
            losses = {}
            target_start = 0
            for target_name, target in targets.items():
                # Get batch targets for target champions for the last timestep
                y = x[f'target_{target_name}'][torch.arange(batch_size), -1, target_idx]
                target_prediction = pred[:, target_start: target_start + target['size']].squeeze()
                losses[target_name] = target['loss'](target_prediction, y) * target['weight']

                test_losses[target_name] += losses[
                                                 target_name].item() * batch_size  # Multiply by # of samples in the batch
                if target_name in test_accuracies:
                    test_accuracies[target_name] += (target_prediction.argmax(1) == y).sum()

                target_start += target['size']
        # Average over samples
        test_losses = {key: value / len(test_dataset) for key, value in test_losses.items()}

        # Divide successes by number of samples
        test_accuracies = {key: value / len(test_dataset) for key, value in test_accuracies.items()}

        for key in test_losses.keys():
            writer.add_scalar(f'Loss/test/{key}', test_losses[key], best_epoch)

        for key in test_accuracies.keys():
            writer.add_scalar(f'Accuracy/test/{key}', test_accuracies[key], best_epoch)

        total_test_loss = sum(test_losses.values())
        writer.add_scalar('Loss/test/total', total_test_loss, best_epoch)

        # Print info to stdout
        print(f'Epoch {best_epoch}:')
        test_losses = {key: round(value, 3) for key, value in test_losses.items()}
        print(f'Test loss: {total_test_loss:.3f} {test_losses}')



@torch.no_grad()
def get_loss_distribution(model: torch.nn.Module,
                          test_dataset: MacroDataset,
                          device: torch.device,
                          targets: dict,
                          batch_size: int,
                          output_folder: str) -> None:
    model.to(device)
    deviations = {target_name: [] for target_name in targets.keys()}
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    for x, target_idx in tqdm(dataloader):
        batch_size = len(target_idx)
        # Move to device
        for key, value in x.items():
            x[key] = value.to(device)

        pred = model(x, target_idx)
        target_start = 0
        for target_name, target in targets.items():
            # Get batch targets for target champions for the last timestep
            y = x[f'target_{target_name}'][torch.arange(batch_size), -1, target_idx]
            target_prediction = pred[:, target_start: target_start + target['size']].squeeze()
            if target_name == 'position':
                deviations[target_name].extend(target_prediction.softmax(dim=1)[torch.arange(batch_size), y].cpu().numpy())
            else:
                deviations[target_name].extend((target_prediction - y).abs().cpu().numpy())

            target_start += target['size']
    for target_name, values in deviations.items():
        plt.hist(values, bins=100)
        plt.title(target_name)
        plt.savefig(os.path.join(output_folder, f'deviation_{target_name}.png'))
        plt.close()





@torch.no_grad()
def visualize_predictions(model: torch.nn.Module,
                          dataset: MacroDataset,
                          targets: dict,
                          map_split_n: int,
                          output_folder: str,
                          feature_mean: dict,
                          feature_std: dict,
                          data_dragon: DataDragon,
                          game_id: int,
                          fow_model: bool) -> None:
    position_prob_alpha = 0.8
    output_image_size = (1000, 1000)  # height, width
    champion_icon_image_size = (40, 40)
    team_colors = (255, 0, 0), (0, 0, 255)  # blue, red in opencv bgr format
    prediction_color = (0, 255, 0)
    ground_truth_color = (0, 0, 255) if fow_model else (0, 255, 255)

    
    # Prepare champion icons for game state visualization
    champion_names = data_dragon.get_champion_names()
    champion_imgs_dict = data_dragon.get_champion_images()
    champion_icons = np.array([cv2.resize(champion_imgs_dict[champion_name], champion_icon_image_size)
                               for champion_name in champion_names])
    map_img = cv2.resize(data_dragon.get_map_image(), output_image_size)
    position_predictions_folder = os.path.join(output_folder, 'position_predictions',str(game_id))
    position_bad_folder = os.path.join(output_folder, 'position_bad_decisions',str(game_id))
    for folder in (position_predictions_folder, position_bad_folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
    max_deviations = {target_name: {'deviation': 0} for target_name in targets.keys()}

    model.to(torch.device('cpu'))
    
    #TODO MAKE FUNC LATER and combine datasets

    try:
        index = dataset.game_name.index(game_id)
        print(index)
    except ValueError:
        print(f"{id} not found in game_names")
    game_idx = index
    
    champ_state_x_idx, champ_state_y_idx = (32, 33) if fow_model else (31, 32)

    
    pos_idx = [champ_state_x_idx, champ_state_y_idx]
    print('dataset.fame_names',dataset.game_name)
    # this starts 
    
    previous_indices = sum(dataset.game_lengths[:game_idx]) * 10
    previous_positions = None
    previous_predicted_positions = []
    # this starts at 4200 and goes to 
    for i in tqdm(range(previous_indices, previous_indices + dataset.game_lengths[game_idx] * 10)):
        x, target_idx = dataset[i]
        x = {key: value.unsqueeze(0) for key, value in x.items()}
        if i % 10 == 0:
            positions = (x['champion_state'][0, -1, :, pos_idx] *
                         feature_std['champion_state'][pos_idx] +
                         feature_mean['champion_state'][pos_idx]) *\
                           np.array(output_image_size) / MAP_SIZE  # Scale to output image size
            positions = positions.round().int()
            positions[:, 1] = output_image_size[0] - positions[:, 1]
            if previous_positions is not None:
                # Visualize all champion movements at once
                canvas = np.copy(map_img)
                # Plot champion icons
                for champ_idx in range(10):
                    team = champ_idx // 5
                    champ_icon = champion_icons[x['champion_types'][0, -1, champ_idx]]
                    pos_x, pos_y = previous_positions[champ_idx, 0].item(), previous_positions[champ_idx, 1].item()
                    champ_color = team_colors[team]
                    # Convert to int icon starts
                    start_x, start_y = round(pos_x - champ_icon.shape[0] // 2), round(pos_y - champ_icon.shape[1] // 2)

                    # Ensure start_x and start_y are not negative
                    start_x = max(0, start_x)
                    start_y = max(0, start_y)

                    # Adjust end_x and end_y to not exceed canvas dimensions
                    end_x = min(canvas.shape[1], start_x + champ_icon.shape[0])
                    end_y = min(canvas.shape[0], start_y + champ_icon.shape[1])

                    # Also adjust champ_icon if necessary
                    if end_x - start_x < champ_icon.shape[0] or end_y - start_y < champ_icon.shape[1]:
                        champ_icon = champ_icon[:end_y-start_y, :end_x-start_x]

                    canvas[start_y:end_y, start_x:end_x] = champ_icon
                    cv2.circle(canvas, (pos_x, pos_y), champion_icon_image_size[0] // 2, champ_color, 2)

                # Plot actual movements
                for champ_idx in range(10):
                    cv2.arrowedLine(canvas,
                                    previous_positions[champ_idx].tolist(),
                                    positions[champ_idx].tolist(),
                                    ground_truth_color,
                                    2,
                                    tipLength=0.05)
                # Plot predicted movements
                for champ_idx in range(10):
                    cv2.arrowedLine(canvas,
                                    previous_positions[champ_idx].tolist(),
                                    previous_predicted_positions[champ_idx].tolist(),
                                    prediction_color,
                                    2,
                                    tipLength=0.05)
                cv2.imwrite(os.path.join(position_predictions_folder, f'{i-10}_movements.png'), canvas)
                previous_predicted_positions.clear()
            previous_positions = positions

        pred = model(x, target_idx)
        target_start = 0
        for target_name, target in targets.items():
            # Get batch targets for target champions for the last timestep
            y = x[f'target_{target_name}'][0, -1, target_idx]
            target_prediction = pred[:, target_start: target_start + target['size']].squeeze()

            if target_name == 'position':
                prediction_map = target_prediction.squeeze().softmax(0).view((map_split_n, map_split_n)).numpy()
                y = y.item()  # Convert from tensor to int
                # Save predicted position for all movements visualization
                predicted_position = ((np.array(np.unravel_index(prediction_map.argmax(), prediction_map.shape))[::-1] + 0.5)
                                      * np.array(output_image_size) / map_split_n).round().astype(int)
                predicted_position[1] = output_image_size[0] - predicted_position[1]
                previous_predicted_positions.append(predicted_position)
                canvas = np.copy(map_img) / 255 * (1 - position_prob_alpha)
                # Add prediction probabilities
                prediction_map_resized = cv2.resize(prediction_map[::-1, :], output_image_size, interpolation=cv2.INTER_NEAREST)
                canvas += position_prob_alpha * np.expand_dims(prediction_map_resized, 2)
                canvas = (canvas * 255).round().astype(np.uint8)

                # Plot champion icons
                for champ_idx in range(10):
                    team = champ_idx // 5
                    champ_state = x['champion_state'][0, -1, champ_idx]
                    champ_icon = champion_icons[x['champion_types'][0, -1, champ_idx]]
                    # Unnormalize position features
                    pos_x, pos_y = (champ_state[pos_idx] * feature_std['champion_state'][pos_idx] +\
                                   feature_mean['champion_state'][pos_idx]) *\
                                   np.array(output_image_size) / MAP_SIZE  # Scale to output image size
                    pos_x, pos_y = round(pos_x.item()), round(pos_y.item())
                    pos_y = output_image_size[0] - pos_y
                    if champ_idx == target_idx:
                        target_position = (pos_x, pos_y)  # Save position for arrow
                        champ_color = prediction_color
                    else:
                        champ_color = team_colors[team]
                    # Convert to int icon starts
                    start_x, start_y = round(pos_x - champ_icon.shape[0] // 2), round(pos_y - champ_icon.shape[1] // 2)
                    x_low, x_high = max(start_x, 0), min(start_x + champ_icon.shape[0], output_image_size[1])
                    y_low, y_high = max(start_y, 0), min(start_y + champ_icon.shape[1], output_image_size[0])
                    canvas[y_low: y_high,
                            x_low: x_high] = champ_icon[:y_high - y_low, :x_high - x_low, :]
                    cv2.circle(canvas, (pos_x, pos_y), champion_icon_image_size[0] // 2, champ_color, 2)
                # Mark the correct prediction target
                sector = np.array([y % map_split_n, y // map_split_n], dtype=int)
                circle_x, circle_y = ((sector + 0.5) * np.array(output_image_size) / map_split_n).round().astype(
                    int)
                circle_y = output_image_size[0] - circle_y
                cv2.arrowedLine(canvas, target_position, (circle_x, circle_y), ground_truth_color, 2, tipLength=0.05)
                cv2.imwrite(os.path.join(position_predictions_folder, f'{i}.png'), canvas)
                if prediction_map[sector[0], sector[1]] < 0.01:
                    cv2.imwrite(os.path.join(position_bad_folder, f'{i}.png'), canvas)
                plt.close()
            else:
                deviation = max((target_prediction - y).item(), 0)
                if deviation > max_deviations[target_name]['deviation']:
                    max_deviations[target_name]['deviation'] = deviation
                    max_deviations[target_name]['value'] = y
                    max_deviations[target_name]['predicted'] = target_prediction
                    max_deviations[target_name]['id'] = i
            target_start += target['size']
    print(max_deviations)
    return position_predictions_folder


def get_filenames(dataset_folder: str, dataset_name: str,fow_model =True) -> List[str]:
    with open(os.path.join(os.path.dirname(__file__),'Game_data', 'macro_prediction_split_fow' if fow_model else 'macro_prediction_split_muller', f'{dataset_name}.txt'), 'r') as f:
        return [os.path.join(dataset_folder, filename.strip()) for filename in f]



def prediction_visualization(game_id,fow_model=True,number_of_proababilities=5):
    
    probabilities = get_log_reg(game_id,fow_model)
    diviation_moments = get_max_deviations(probabilities,number_of_proababilities)
    
    data_folder = os.path.join(os.path.dirname(__file__),'Game_data')
    POSITON_PREDICTION_SPLIT_N = 12
    history_size = 1
    
    dataset_folder = os.path.join(data_folder, 'macro_fow' if fow_model else 'macro_dataset')
    models_folder = 'models'
    
    if not os.path.exists(models_folder):
        os.makedirs(models_folder, exist_ok=True)
    output_folder = os.path.join(os.path.dirname(__file__),'output', 'macro_prediction_fow' if fow_model else 'macro_prediction_muller')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    version = '13.4'
    data_dragon = DataDragon(data_folder, version)
    item_data = data_dragon.get_item_data()
    champion_data = data_dragon.get_champion_data()
    summoner_data = data_dragon.get_summoner_data()

    print("Loading train dataset to memory")
    train_dataset = MacroDataset(get_filenames(dataset_folder, 'train'), history_size)
    print("Loading valid dataset to memory")
    valid_dataset = MacroDataset(get_filenames(dataset_folder, 'valid'), history_size)
    print("Loading test dataset to memory")
    test_dataset = MacroDataset(get_filenames(dataset_folder, 'test'), history_size)

    

    # Get mean and std of each feature and normalize datasets so that mean=0 and std=1 for each feature
    feature_mean, feature_std = train_dataset.get_feature_mean_std()
    for key, value in feature_std.items():
        zero = value == 0
        if zero.any():
            value[zero] = 1
    train_dataset.normalize(feature_mean, feature_std)
    valid_dataset.normalize(feature_mean, feature_std)
    test_dataset.normalize(feature_mean, feature_std)

    target_mean, target_std = train_dataset.get_target_mean_std()

    champion_skill_names = [spell['id'].lower()
                            for champion, data in champion_data['data'].items() for spell in data['spells']]

    eg, _ = train_dataset[0]

    features = [
        'state',
        # 'stats', causes overfitting
        'gameTime',
        'map',
        'champion_state',
        'champion_embedding',
        'items',
        'wards',
        'summoner_spells',
        'skills'
    ]

    target_names = ['position', 'minions', 'monsters', 'kills', 'turrets', 'inhibitors', 'heralds', 'dragons', 'barons']
    targets = {target: {'size': eg[f'target_{target}'].shape[-1] // 10 if target != 'position' else POSITON_PREDICTION_SPLIT_N ** 2,
                        'loss': torch.nn.MSELoss() if target != 'position' else torch.nn.CrossEntropyLoss(),
                        'weight': 1 / target_std[f'target_{target}'].pow(2).to(device) if target != 'position' else 1}
               for target in target_names}
    feature_sizes = {'state': eg['state'].shape[1],
                     'stats': eg['stats'].shape[1],
                     'gameTime': eg['gameTime'].shape[1],
                     'map': eg['map'].shape[1:],
                     'champion_state': eg['champion_state'].shape[2],
                     'champion_stats': eg['champion_stats'].shape[2],
                     'champion_embedding': len(champion_data['data']) + 1,
                     'items': eg['items'].shape[3],
                     'item_embedding': len(item_data['data']) + 1,
                     'skill_embedding': len(champion_skill_names) + 1,
                     'summoner_spell_embedding': len(summoner_data['data']) + len(EXTRA_SUMMONER_SPELL_NAMES) + 1,
                     'wards': {ward_type: eg[f'{ward_type}_wards'].shape[3]
                               for ward_type in WARD_TYPES},
                     'summoner_spells': eg['summoner_spells'].shape[3],
                     'skills': eg['skills'].shape[3]}
    target_size = sum(target['size'] for target in targets.values())

    # Network hyperparameters
    embedding_sizes = {'champion_embedding': 2,
                       'item_embedding': 2,
                       'summoner_spell': 2,
                       'skill': 2}
    hidden_sizes = {'map': 8,
                    'items': 2,
                    'wards': {ward_type: 2 for ward_type in WARD_TYPES},
                    'summoner_spells': 2,
                    'skills': 2,
                    'champion': 32,
                    'encoder': 128,
                    'prediction_layers': 1}

    batch_size = 64
    model_name = 'Best'
    
    macro_predictor = create_model(device,
                                   features,
                                   targets,
                                   feature_sizes,
                                   target_size,
                                   embedding_sizes,
                                   hidden_sizes,
                                   history_size,
                                   model_name)
    model_path = os.path.join(os.path.dirname(__file__),'prediction_models', 'macro_predictor_fow.pt' if fow_model else 'macro_predictor.pt' )

    macro_predictor.load_state_dict(torch.load(model_path))
    macro_predictor.eval()
    prediction_folder = visualize_predictions(macro_predictor,
                          test_dataset,
                          targets,
                          POSITON_PREDICTION_SPLIT_N,
                          output_folder,
                          feature_mean,
                          feature_std,
                          data_dragon,
                          game_id=game_id,
                          fow_model=fow_model)
    return prediction_folder , diviation_moments



def main():
    if len(sys.argv) == 1:
        print("No arguments were provided.")
    else:
        game_id, fow_model = sys.argv[1], sys.argv[2]
    prediction_visualization(game_id = 3120261,fow_model=True)

if __name__ == "__main__":
    main()


