from videogen.data.steamboat_willie.dataset import SteamboatWillieDataset
from videogen.config import Config


def get_dataset(config: Config, mode: str):
    if config.dataset.name == 'steamboat-willie':
        return SteamboatWillieDataset(config, mode)