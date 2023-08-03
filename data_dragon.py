import cv2
import json
import numpy as np
import os
from typing import List

from lol_rules import EXTRA_SUMMONER_SPELL_NAMES


class DataDragon:
    data_folder: str
    img_folder: str

    def __init__(self, data_folder: str, version: str):
        language = 'en_US'
        dragontail_version = version + '.1'
        dragontail_folder = os.path.join(data_folder, f'dragontail-{dragontail_version}')
        self.data_folder = os.path.join(dragontail_folder, dragontail_version, 'data', language)
        self.img_folder = os.path.join(dragontail_folder, dragontail_version, 'img')

    def get_item_data(self) -> dict:
        with open(os.path.join(self.data_folder, 'item.json'), 'r') as f:
            return json.load(f)

    def get_champion_data(self) -> dict:
        with open(os.path.join(self.data_folder, 'championFull.json'), 'r', encoding='utf8') as f:
            return json.load(f)

    def get_rune_data(self) -> dict:
        with open(os.path.join(self.data_folder, 'runesReforged.json'), 'r') as f:
            return json.load(f)

    def get_summoner_data(self) -> dict:
        with open(os.path.join(self.data_folder, 'summoner.json'), 'r') as f:
            return json.load(f)

    def get_champion_images(self) -> dict:
        champion_img_folder = os.path.join(self.img_folder, 'champion')
        champion_imgs = {}
        for filename in os.listdir(champion_img_folder):
            champion_name = os.path.splitext(filename)[0].lower()
            champion_imgs[champion_name] = cv2.imread(os.path.join(champion_img_folder, filename))
        return champion_imgs

    def get_champion_names(self) -> List[str]:
        return list(sorted(map(str.lower, self.get_champion_data()['data'].keys())))

    def get_item_names(self) -> List[str]:
        item_data = self.get_item_data()['data']
        return [item_data[item]['name'] for item in sorted(item_data.keys())]

    def get_summoner_spell_names(self) -> List[str]:
        return list(sorted(map(str.lower, self.get_summoner_data()['data'].keys()))) + EXTRA_SUMMONER_SPELL_NAMES

    def get_champion_skill_names(self) -> List[str]:
        champion_data = self.get_champion_data()
        champion_skill_names = {champion.lower(): [spell['id'].lower() for spell in data['spells']]
                                for champion, data in champion_data['data'].items()}
        return [skill for skill_set in champion_skill_names.values() for skill in skill_set]

    def get_map_image(self) -> np.ndarray:
        return cv2.imread(os.path.join(self.img_folder, 'map', 'map11.png'))
