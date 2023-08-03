import torch
from torch import nn, Tensor
from typing import Optional, Sequence, Union, Tuple

from lol_rules import WARD_TYPES


class MacroPredictor(nn.Module):
    features: Sequence[str]
    targets: Sequence[str]
    target_sizes: dict
    target_types: dict
    champion_embedding_size: int
    item_embedding_size: int
    summoner_spell_embedding_size: int
    skill_embedding_size: int
    map_shape: Tuple[int, int, int]
    map_processor: Optional[nn.Sequential]
    champion_embedding: Optional[nn.Embedding]
    item_embedding: Optional[nn.Embedding]
    item_set_processor: Optional[nn.Sequential]
    control_ward_set_processor: Optional[nn.Sequential]
    sight_ward_set_processor: Optional[nn.Sequential]
    farsight_ward_set_processor: Optional[nn.Sequential]
    summoner_spell_embedding: Optional[nn.Embedding]
    summoner_spell_set_processor: Optional[nn.Sequential]
    skill_embedding: Optional[nn.Embedding]
    skill_set_processor: Optional[nn.Sequential]
    champion_set_processor: nn.Sequential
    encoder: Union[nn.LSTM, nn.Sequential]
    prediction_head: Union[nn.Sequential, nn.Linear]

    def __init__(self,
                 features: Sequence[str],
                 targets: Sequence[str],
                 feature_sizes: dict,
                 target_size: int,
                 embedding_sizes: dict,
                 hidden_sizes: dict,
                 history_size: int):
        super().__init__()
        assert len(set(features)) == len(features) and len(set(targets)) == len(targets)
        self.features, self.targets = features, targets
        self.champion_embedding_size = embedding_sizes.get('champion_embedding', 0)
        self.item_embedding_size = embedding_sizes.get('item_embedding', 0)
        self.summoner_spell_embedding_size = embedding_sizes.get('summoner_spell', 0)
        self.skill_embedding_size = embedding_sizes.get('skill', 0)
        champion_size = 0
        state_size = 0
        self.target_size = target_size

        self.map_processor = None
        self.champion_embedding = None
        self.item_embedding = None
        self.item_set_processor = None
        self.control_ward_set_processor = None
        self.sight_ward_set_processor = None
        self.farsight_ward_set_processor = None
        self.summoner_spell_embedding = None
        self.summoner_spell_set_processor = None
        self.skill_embedding = None
        self.skill_set_processor = None

        for feature in features:
            feature_size = feature_sizes[feature]
            hidden_size = hidden_sizes.get(feature, 0)
            if feature in ('state', 'stats', 'gameTime'):  # Global stats, not processed in any way
                state_size += feature_size
            elif feature in ('champion_stats', 'champion_state'):
                champion_size += feature_size
            elif feature == 'map':
                self.map_shape = feature_size
                channels, width, height = feature_size
                self.map_processor = nn.Sequential(
                    nn.Conv2d(in_channels=channels, out_channels=hidden_size, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.MaxPool2d(2),
                    nn.LeakyReLU(),
                    nn.Conv2d(in_channels=hidden_size, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.MaxPool2d(2),
                    nn.LeakyReLU(),
                    nn.Flatten()
                )
                state_size += height // 4 * width // 4
            elif feature == 'champion_embedding':
                self.champion_embedding = nn.Embedding(feature_sizes[feature], self.champion_embedding_size)
                champion_size += self.champion_embedding_size
            elif feature == 'items':
                self.item_embedding = nn.Embedding(feature_sizes['item_embedding'], self.item_embedding_size)
                self.item_set_processor = nn.Sequential(
                    nn.Linear(feature_size + self.item_embedding_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU()
                )
                champion_size += hidden_size
            elif feature == 'wards':
                self.control_ward_set_processor = nn.Sequential(
                    nn.Linear(feature_size['control'], hidden_size['control']),
                    nn.ReLU(),
                    nn.Linear(hidden_size['control'], hidden_size['control']),
                    nn.ReLU()
                )
                self.sight_ward_set_processor = nn.Sequential(
                    nn.Linear(feature_size['sight'], hidden_size['sight']),
                    nn.ReLU(),
                    nn.Linear(hidden_size['sight'], hidden_size['sight']),
                    nn.ReLU()
                )
                self.farsight_ward_set_processor = nn.Sequential(
                    nn.Linear(feature_size['farsight'], hidden_size['farsight']),
                    nn.ReLU(),
                    nn.Linear(hidden_size['farsight'], hidden_size['farsight']),
                    nn.ReLU()
                )
                champion_size += sum(hidden_size[ward_type] for ward_type in WARD_TYPES)
            elif feature == 'summoner_spells':
                self.summoner_spell_embedding = nn.Embedding(feature_sizes['summoner_spell_embedding'],
                                                             self.summoner_spell_embedding_size)
                self.summoner_spell_set_processor = nn.Sequential(
                    nn.Linear(feature_size + self.summoner_spell_embedding_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU()
                )
                champion_size += hidden_size
            elif feature == 'skills':
                self.skill_embedding = nn.Embedding(feature_sizes['skill_embedding'], self.skill_embedding_size)
                self.skill_set_processor = nn.Sequential(
                    nn.Linear(feature_size + self.skill_embedding_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU()
                )
                champion_size += hidden_size
            else:
                raise NotImplementedError
        self.champion_set_processor = nn.Sequential(
            nn.Linear(champion_size, hidden_sizes['champion']),
            nn.ReLU(),
            nn.Linear(hidden_sizes['champion'], hidden_sizes['champion']),
            nn.ReLU()
        )
        if history_size == 1:
            self.encoder = nn.Sequential(
                nn.Linear(state_size + hidden_sizes['champion'] * 2, hidden_sizes['encoder']),
                nn.ReLU()
            )
            self.prediction_head = nn.Linear(hidden_sizes['encoder'], self.target_size)
        else:
            self.encoder = nn.LSTM(state_size + hidden_sizes['champion'] * 2,
                                   hidden_sizes['encoder'],
                                   num_layers=hidden_sizes['prediction_layers'],
                                   batch_first=True)
            self.prediction_head = nn.Sequential(nn.Linear(hidden_sizes['encoder'], self.target_size))

    def forward(self, src: dict, champion_idx: Tensor) -> Tensor:
        state = self.get_state(src, champion_idx)
        if isinstance(self.encoder, nn.LSTM):
            state, _ = self.encoder(state)
            state = state[:, -1, :]
        else:  # Linear layer encoder
            state = self.encoder(state.squeeze(1))
        targets = self.prediction_head(state)
        return targets

    def get_state(self, src: dict, champion_idx: Tensor) -> Tensor:
        batch_size, history_size = next(iter(src.values())).shape[:2]
        features, champion_features = [], []
        if 'state' in self.features:
            features.append(src['state'])
        if 'stats' in self.features:
            features.append(src['stats'])
        if 'gameTime' in self.features:
            features.append(src['gameTime'])
        if 'champion_state' in self.features:
            champion_features.append(src['champion_state'])
        if 'champion_stats' in self.features:
            champion_features.append(src['champion_stats'])
        if 'map' in self.features:
            map_features = self.map_processor(src['map'].view((-1,) + self.map_shape))
            map_features = map_features.view(batch_size, history_size, -1)
            features.append(map_features)
        if 'champion_embedding' in self.features:
            champion_features.append(self.champion_embedding(src['champion_types']))
        if 'items' in self.features:
            item_embeddings = self.item_embedding(src['item_types'])
            processed_items = self.item_set_processor(torch.concat((item_embeddings, src['items']), dim=-1))
            item_features, _ = processed_items.max(dim=-2)  # max pool over items
            champion_features.append(item_features)
        if 'wards' in self.features:
            control_processed = self.control_ward_set_processor(src['control_wards'])
            sight_processed = self.sight_ward_set_processor(src['sight_wards'])
            farsight_processed = self.farsight_ward_set_processor(src['farsight_wards'])
            control_features, _ = control_processed.max(dim=-2)
            sight_features, _ = sight_processed.max(dim=-2)
            farsight_features, _ = farsight_processed.max(dim=-2)
            champion_features.extend((control_features, sight_features, farsight_features))
        if 'summoner_spells' in self.features:
            summoner_spell_embeddings = self.summoner_spell_embedding(src['summoner_spell_types'])
            processed_summoner_spells = self.summoner_spell_set_processor(
                torch.concat((summoner_spell_embeddings, src['summoner_spells']), dim=-1))
            summoner_spell_features, _ = processed_summoner_spells.max(dim=-2)
            champion_features.append(summoner_spell_features)
        if 'skills' in self.features:
            skill_embeddings = self.skill_embedding(src['skill_types'])
            processed_skills = self.skill_set_processor(torch.concat((skill_embeddings, src['skills']), dim=-1))
            skill_features, _ = processed_skills.max(dim=-2)
            champion_features.append(skill_features)
        champion_features = torch.concat(champion_features, dim=-1)
        champion_embeddings = self.champion_set_processor(champion_features)
        champion_features, _ = champion_embeddings.max(dim=-2)
        features.append(champion_features)
        features.append(champion_embeddings[torch.arange(batch_size), :, champion_idx, :])
        return torch.concat(features, dim=-1)
