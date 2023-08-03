import numpy as np

MAP_SIZE = 15000

WARD_DURATIONS = {'control': np.inf,
                  'sight': 150 * 1000,
                  'yellowTrinket': lambda avg_level: (90 + 30 / 17 * (avg_level - 1)) * 1000,
                  'blueTrinket': np.inf}
# Maps raw ward types to classes which share limits
WARD_TYPE_MAP = {'control': 'control',
                 'sight': 'sight',
                 'yellowTrinket': 'sight',
                 'blueTrinket': 'farsight'}
WARD_LIMITS = {'control': 1,
               'sight': 3,
               'farsight': 20}

LANES = ['top', 'mid', 'bot']
DRAGON_TYPES = ['air', 'earth', 'water', 'fire', 'elder','chemtech','hextech']
CHAMPION_ROLES = ['top', 'mid', 'support', 'adc', 'jungle']
WARD_TYPES = ['control', 'sight', 'farsight']
TEAM_EPIC_MONSTERS = ['raptor', 'wolf', 'gromp', 'krug', 'redCamp', 'blueCamp']
NEUTRAL_EPIC_MONSTERS = ['scuttleCrab', 'dragon', 'riftHerald', 'baron']

TURRET_TIERS = ['outer', 'inner', 'base']
TURRET_TYPES = [f'turret_{turret_tier}_{lane}' for lane in LANES for turret_tier in TURRET_TIERS] + ['turret_nexus_mid']

EXTRA_SUMMONER_SPELL_NAMES = ['s5_summonersmiteduel', 's5_summonersmiteplayerganker', 'summonerflashperkshextechflashtraptionv2','s12_summonerteleportupgrade','summonersmiteavatardefensive','summonersmiteavataroffensive','summonersmiteavatarutility']

# Times in ms to match game time format
BARON_BUFF_DURATION = 180 * 1000
DRAGON_BUFF_DURATION = 150 * 1000
CAMP_BUFF_DURATIONS = 120 * 1000
INHIBITOR_RESPAWN_TIME = 5 * 60 * 1000
SPAWN_TIMES = {'minions': 65 * 1000,
               'raptor': 90 * 1000,
               'wolf': 90 * 1000,
               'krug': 102 * 1000,
               'gromp': 102 * 1000,
               'blueCamp': 90 * 1000,
               'redCamp': 90 * 1000,
               'scuttleCrab': 195 * 1000,
               'dragon': 5 * 60 * 1000,
               'riftHerald': 8 * 60 * 1000,
               'baron': 20 * 60 * 1000}
RESPAWN_TIMES = {'minions': 30 * 1000,
                 'raptor': 135 * 1000,
                 'wolf': 135 * 1000,
                 'krug': 135 * 1000,
                 'gromp': 135 * 1000,
                 'blueCamp': 5 * 60 * 1000,
                 'redCamp': 5 * 60 * 1000,
                 'scuttleCrab': 150 * 1000,
                 'dragon': 5 * 60 * 1000,
                 'riftHerald': 6 * 60 * 1000,
                 'baron': 6 * 60 * 1000}
