from typing import Optional
import warnings
from pathlib import Path
import os

import pandas as pd

warnings.simplefilter(action="ignore", category=FutureWarning)
# The intention is to remove this warning:
#
# /env/lib/python3.8/site-packages/pandas/core/frame.py:1482: FutureWarning: Using short name for 'orient' is deprecated. Only the options: ('dict', list, 'series', 'split', 'records', 'index') will be used in a future version. Use one of the above to silence this warning.
#  warnings.warn(

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
# your performance may suffer as PyTables will pickle object types that it cannot


def create_directory_if_not_exists(path: Path) -> None:
    if not path.is_dir():
        try:
            path.mkdir()
        except Exception as e:
            print(e)


DIR_MODULE_SUB = Path(__file__).resolve().parent  # ../geoguessr/geoguessr

# ../geoguessr/geoguessr/data
DIR_DATA = DIR_MODULE_SUB.joinpath("data")

# ../geoguessr/geoguessr/data/model
DIR_MODEL = DIR_DATA.joinpath("model")

# ../geoguessr/geoguessr/data/plots
DIR_PLOTS = DIR_DATA.joinpath("plots")

# ../geoguessr/geoguessr/data/plots
DIR_HISTORY = DIR_DATA.joinpath("history")

# ../geoguessr/geoguessr/data/raw
DIR_RAW = DIR_DATA.joinpath("raw")

# ../geoguessr/geoguessr/data/uploads
DIR_UPLOADS = DIR_DATA.joinpath("uploads")

# ../geoguessr/geoguessr/data/raw
DIR_MODELS_PROD = DIR_MODULE_SUB.joinpath("models")




create_directory_if_not_exists(DIR_DATA)
create_directory_if_not_exists(DIR_MODEL)
create_directory_if_not_exists(DIR_PLOTS)
create_directory_if_not_exists(DIR_HISTORY)

pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 25)
pd.options.display.float_format = "{:.2f}".format
pd.options.mode.chained_assignment = None


