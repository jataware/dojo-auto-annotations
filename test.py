from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import xarray as xr

from MetadataSchema import (
    AnnotationSchema,
    GeoAnnotation,
    DateAnnotation,
    FeatureAnnotation,
    ColumnType,
    DateType,
    GeoType,
    FeatureType,
    CoordFormat,
    GadmLevel,
    TimeField,
    LatLong,
    TimeRange,
)


from agent import Agent, set_openai_key


import sys
import pdb


def custom_except_hook(exctype, value, traceback):
    if exctype == AssertionError:
        print(f'AssertionError: {value}')
        pdb.post_mortem(traceback)
    else:
        # Call the default excepthook to handle other exceptions
        sys.__excepthook__(exctype, value, traceback)


# Set the custom exception hook
sys.excepthook = custom_except_hook


def main():
    set_openai_key()

    meta = get_meta()
    meta = [meta[2]]

    # shorten the description if necessary
    agent = Agent(model='gpt-4-turbo-preview')

    for m in meta:
        if len(m.description) > 1000:
            m.description = shorten_description(m, agent)

        if m.path.suffix == '.csv':
            handle_csv(m, agent)
        # elif m.path.suffix == '.xlsx':
        #     handle_xlsx(m, agent)
        # elif m.path.suffix == '.nc':
        #     handle_netcdf(m, agent)
        # elif m.path.suffix == '.tif' or m.path.suffix == '.tiff':
        #     handle_geotiff(m, agent)
        else:
            raise ValueError(f'Unhandled file type: {m.path.suffix}')


def shorten_description(meta: Meta, agent: Agent) -> str:
    desc = agent.oneshot_sync('You are a helpful assistant.', f'''\
I have a dataset called "{meta.name}" With the following description:
"{meta.description}"
If there is a lot of superfluous information, could you pare it down to just the key details? Output only the new description without any other comments. If there are not superfluous details, output only the original unmodified description.\
''')
    return desc


def enum_to_keys(enum):
    return [e.name for e in enum]


def handle_csv(meta: Meta, agent: Agent) -> AnnotationSchema:
    df = pd.read_csv(meta.path)

    # # identify the type of each column
    # geo_columns:list[GeoAnnotation] = []
    # date_columns:list[DateAnnotation] = []
    # feature_columns:list[FeatureAnnotation] = []

    # map from all ColumnType keys to empty lists
    column_type_map = {col_type.name: [] for col_type in ColumnType}

    for col in df.columns:
        # ask the model to identify the type of the column
        col_type = agent.oneshot_sync('You are a helpful assistant', f'''\
I have a dataset called "{meta.name}" with the following description:
"{meta.description}"
I have a column called "{col}" with the following values (first 5 rows):
{df[col].head().to_string()}
I need to determine if this column contains geographic information, date/time information, or feature information.
Please answer either {', '.join(enum_to_keys(ColumnType))}, or UNSURE without any other comments.\
''')
        assert col_type in column_type_map or col_type == 'UNSURE', f'LLM returned an invalid column type: `{col_type}`'

        if col_type == 'UNSURE':
            col_type = ask_user(f'''\
    The LLM was unsure about the column type for {col} with the following values (first 5 rows):
    {df[col].head().to_string()}
    What type of data is this? ({', '.join(enum_to_keys(ColumnType))}) \
''')
            assert col_type in column_type_map, f'invalid column type: `{col_type}`'
        else:
            print(f'LLM identified column "{col}" as a {col_type}')

        column_type_map[col_type].append(col)

    ##### determine the type of date column for each #####
    date_type_map = {date_type.name: [] for date_type in DateType}

    for col in column_type_map['DATE']:
        date_type = agent.oneshot_sync('You are a helpful assistant', f'''\
I have a dataset called "{meta.name}" with the following description:
"{meta.description}"
I have a column called "{col}" with the following values (first 5 rows):
{df[col].head().to_string()}
The column has been identified as containing date/time information.
I need to identify the type of date/time information it contains.
Please answer either {', '.join(enum_to_keys(DateType))}, or UNSURE without any other comments.\
''')
        assert date_type in date_type_map or date_type == 'UNSURE', f'LLM returned an invalid date type: `{date_type}`'

        if date_type == 'UNSURE':
            date_type = ask_user(f'''\
The LLM was unsure about the date type for {col} with the following values (first 5 rows):
{df[col].head().to_string()}
What type of date is this? ({', '.join(enum_to_keys(DateType))}) \
''')
            assert date_type in date_type_map, f'invalid date type: `{date_type}`'
        else:
            print(f'LLM identified DATE column "{col}" as a {date_type}')

        date_type_map[date_type].append(col)

    ##### TODO: identifying the primary time column or if they need to be built #####

    ##### identifying the type of geo column for each #####
    geo_type_map = {geo_type.name: [] for geo_type in GeoType}

    for col in column_type_map['GEO']:
        geo_type = agent.oneshot_sync('You are a helpful assistant', f'''\
I have a dataset called "{meta.name}" with the following description:
"{meta.description}"
I have a column called "{col}" with the following values (first 5 rows):
{df[col].head().to_string()}
The column has been identified as containing geographic information.
I need to identify the type of geographic information it contains.
Please answer either {', '.join(enum_to_keys(GeoType))}, or UNSURE without any other comments.\
''')
        assert geo_type in geo_type_map or geo_type == 'UNSURE', f'LLM returned an invalid geo type: `{geo_type}`'

        if geo_type == 'UNSURE':
            geo_type = ask_user(f'''\
The LLM was unsure about the geo type for {col} with the following values (first 5 rows):
{df[col].head().to_string()}
What type of geo data is this? ({', '.join(enum_to_keys(GeoType))}) \
''')
            assert geo_type in geo_type_map, f'invalid geo type: `{geo_type}`'
        else:
            print(f'LLM identified GEO column "{col}" as a {geo_type}')

        geo_type_map[geo_type].append(col)

    ##### identifying the type of feature column for each #####
    feature_type_map = {feature_type.name: [] for feature_type in FeatureType}

    for col in column_type_map['FEATURE']:
        feature_type = agent.oneshot_sync('You are a helpful assistant', f'''\
I have a dataset called "{meta.name}" with the following description:
"{meta.description}"
I have a column called "{col}" with the following values (first 5 rows):
{df[col].head().to_string()}
The column has been identified as containing feature information.
I need to identify the type of feature information it contains.
Please answer either {', '.join(enum_to_keys(FeatureType))}, or UNSURE without any other comments.\
''')
        assert feature_type in feature_type_map or feature_type == 'UNSURE', f'LLM returned an invalid feature type: `{feature_type}`'

        if feature_type == 'UNSURE':
            feature_type = ask_user(f'''\
The LLM was unsure about the feature type for {col} with the following values (first 5 rows):
{df[col].head().to_string()}
What type of feature data is this? ({', '.join(enum_to_keys(FeatureType))}) \
''')
            assert feature_type in feature_type_map, f'invalid feature type: `{feature_type}`'
        else:
            print(f'LLM identified FEATURE column "{col}" as a {feature_type}')

        feature_type_map[feature_type].append(col)

    pdb.set_trace()


def handle_netcdf(meta: Meta, agent: Agent) -> AnnotationSchema:
    data = xr.open_dataset(meta.path)
    pdb.set_trace()


def handle_geotiff(meta: Meta, agent: Agent) -> AnnotationSchema:
    data = xr.open_rasterio(meta.path)
    pdb.set_trace()


def handle_xlsx(meta: Meta, agent: Agent) -> AnnotationSchema:
    df = pd.read_excel(meta.path)
    pdb.set_trace()


# Could be more complicated, e.g. use UI to ask user
def ask_user(prompt: str) -> str:
    return input(prompt)


@dataclass
class Meta:
    path: Path
    name: str
    description: str

    def __init__(self, path: str, name: str, description: str):
        assert path.startswith('[') and path.endswith(']')
        self.path = Path('datasets', path[1:-1])
        assert name.startswith('Name:')
        self.name = name[5:].strip()
        assert description.startswith('Description:')
        self.description = description[12:].strip()


def get_meta() -> list[Meta]:
    meta = Path('meta.txt').read_text()
    meta = meta.split('\n\n')
    meta = [m.strip() for m in meta]
    meta = [m for m in meta if m]
    meta = [Meta(*m.split('\n')) for m in meta]

    return meta


if __name__ == '__main__':
    main()
