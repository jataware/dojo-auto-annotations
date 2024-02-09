from __future__ import annotations
from typing import TypeVar

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


from agent import Agent, set_openai_key, Message, Role


import sys
import pdb


# drop me into a pdb context on an assertion error
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
    agent = Agent(model='gpt-4-turbo-preview', timeout=10.0)

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


T = TypeVar('T')


def identify_column_type(agent: Agent, df: pd.DataFrame, col: str, meta: Meta, options: list[T], prompt: str) -> T | None:
    options_or_unsure = options + ['UNSURE']
    options_or_none = options + ['NONE']

    # initial messages to the llm
    messages = [
        Message(Role.system, 'You are a helpful assistant.'),
        Message(Role.user, f'''\
I have a dataset called "{meta.name}" with the following description:
"{meta.description}"
I have a column called "{col}" with the following values (first 5 rows):
{df[col].head().to_string()}
{prompt}
Please select one of the following options: {', '.join(options)}, or UNSURE. Write your answer without any other comments.\
'''
                )
    ]

    # ask the model to identify the type of the column
    res = agent.multishot_sync(messages)

    # reprompt the LLM if it didn't give a valid answer
    if res not in options_or_unsure:
        messages.append(Message(Role.assistant, res))
        messages.append(Message(
            Role.system, f'`{res}` is not a valid answer. Please select one of the following options: {", ".join(options)}, or UNSURE. Write your answer without any other comments.'))
        res = agent.multishot_sync(messages)

    # if it failed a second time, just set it to UNSURE
    if res not in options_or_unsure:
        res = 'UNSURE'

    # have the user fill in the answer if the LLM was unsure or failed twice
    while res not in options_or_none:
        res = ask_user(f'''\
The LLM was unsure about the date type for "{col}" with the following values (first 5 rows):
{df[col].head().to_string()}
{prompt=}
Select one of the following options: {', '.join(options)} or None: \
''')
        res = res.upper()
        if res not in options_or_none:
            print(f'invalid option: `{res}` out of {options=}')

    if res == 'NONE':
        return None
    return res


def enum_to_keys(enum):
    return [e.name for e in enum]


def handle_csv(meta: Meta, agent: Agent) -> AnnotationSchema:
    df = pd.read_csv(meta.path)

    # map from all ColumnType keys to empty lists
    column_type_map = {col_type.name: [] for col_type in ColumnType}

    for col in df.columns:
        col_type = identify_column_type(
            agent, df, col, meta,
            enum_to_keys(ColumnType),
            'I need to determine if this column contains geographic information, date/time information, or feature information.'
        )
        print(f'LLM identified column "{col}" as a {col_type}')
        column_type_map[col_type].append(col)

    ##### determine the type of date column for each #####
    # date_type_map = {date_type.name: [] for date_type in DateType}
    date_type_map = {}

    for col in column_type_map['DATE']:
        date_type = identify_column_type(
            agent, df, col, meta,
            enum_to_keys(DateType),
            '''\
The column has been identified as containing date/time information.
I need to identify the type of date/time information it contains.\
            '''
        )
        print(f'LLM identified DATE column "{col}" as a {date_type}')
        date_type_map[col] = date_type

    ##### TODO: identifying the primary time column or if they need to be built #####

    ##### identifying the type of geo column for each #####
    # geo_type_map = {geo_type.name: [] for geo_type in GeoType}
    geo_type_map = {}

    for col in column_type_map['GEO']:
        geo_type = identify_column_type(
            agent, df, col, meta,
            enum_to_keys(GeoType),
            '''\
The column has been identified as containing geographic information.
I need to identify the type of geographic information it contains.\
            '''
        )
        print(f'LLM identified GEO column "{col}" as a {geo_type}')
        geo_type_map[col] = geo_type

    ##### identifying the type of feature column for each #####
    # feature_type_map = {feature_type.name: [] for feature_type in FeatureType}
    feature_type_map = {}

    for col in column_type_map['FEATURE']:
        feature_type = identify_column_type(
            agent, df, col, meta,
            enum_to_keys(FeatureType),
            '''\
The column has been identified as containing feature information.
I need to identify the type of feature information it contains.\
            '''
        )
        print(f'LLM identified FEATURE column "{col}" as a {feature_type}')
        feature_type_map[col] = feature_type

    geo_annotations: list[GeoAnnotation] = []
    date_annotations: list[DateAnnotation] = []
    feature_annotations: list[FeatureAnnotation] = []

    for col in column_type_map['DATE']:
        if date_type_map[col] is not None:
            date_annotations.append(DateAnnotation(
                name=col,
                display_name=None,
                description=None,
                date_type=DateType[date_type_map[col]].value,
                primary_date=None,
                time_format='todo',  # have the llm figure this part out
                associated_columns=None,
                qualifies=None,
            ))

    for col in column_type_map['GEO']:
        if geo_type_map[col] is not None:
            geo_annotations.append(GeoAnnotation(
                name=col,
                display_name=None,
                description=None,
                geo_type=GeoType[geo_type_map[col]].value,
                primary_geo=None,
                resolve_to_gadm=None,
                is_geo_pair=None,
                coord_format=None,
                qualifies=None,
                gadm_level=None,
            ))

    for col in column_type_map['FEATURE']:
        if feature_type_map[col] is not None:
            feature_annotations.append(FeatureAnnotation(
                name=col,
                display_name=None,
                description='todo feature description',  # need llm to pick this before making the annotation
                feature_type=FeatureType[feature_type_map[col]].value,
                units=None,
                units_description=None,
                qualifies=None,
                qualifierrole=None,
            ))

    # TODO: other processing
    pdb.set_trace()

    return AnnotationSchema(
        geo=geo_annotations,
        date=date_annotations,
        feature=feature_annotations
    )


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
