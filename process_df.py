from __future__ import annotations

from agent import Message, Role, Agent
from meta import Meta
import pandas as pd
from typing import TypeVar
from utils import enum_to_keys, ask_user

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

import pdb


def handle_csv(meta: Meta, agent: Agent) -> AnnotationSchema:
    df = pd.read_csv(meta.path)
    return handle_df(df, meta, agent)


def handle_xlsx(meta: Meta, agent: Agent) -> AnnotationSchema:
    df = pd.read_excel(meta.path)
    return handle_df(df, meta, agent)


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


def identify_column_groups(agent: Agent, df: pd.DataFrame, col: str, meta: Meta, options: list[T], prompt: str) -> T | None:
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
Please select zero or more of the following options: {', '.join(options)}. If you are unsure, only output UNSURE. Write your answer without any other comments.\
'''
                )
    ]

    # ask the model to identify the type of the column
    res = agent.multishot_sync(messages)

    pdb.set_trace()
#     # reprompt the LLM if it didn't give a valid answer
#     if res not in options_or_unsure:
#         messages.append(Message(Role.assistant, res))
#         messages.append(Message(
#             Role.system, f'`{res}` is not a valid answer. Please select one of the following options: {", ".join(options)}, or UNSURE. Write your answer without any other comments.'))
#         res = agent.multishot_sync(messages)

#     # if it failed a second time, just set it to UNSURE
#     if res not in options_or_unsure:
#         res = 'UNSURE'

#     # have the user fill in the answer if the LLM was unsure or failed twice
#     while res not in options_or_none:
#         res = ask_user(f'''\
# The LLM was unsure about the date type for "{col}" with the following values (first 5 rows):
# {df[col].head().to_string()}
# {prompt=}
# Select one of the following options: {', '.join(options)} or None: \
# ''')
#         res = res.upper()
#         if res not in options_or_none:
#             print(f'invalid option: `{res}` out of {options=}')

#     if res == 'NONE':
#         return None
#     return res


def handle_df(df: pd.DataFrame, meta: Meta, agent: Agent) -> AnnotationSchema:
    # map from all ColumnType keys to empty lists
    column_type_map = {col_type.name: [] for col_type in ColumnType}

    for col in df.columns:
        col_type = identify_column_type(
            agent, df, col, meta,
            enum_to_keys(ColumnType),
            'I need to determine if this column contains geographic information, date/time information, or feature information. If it is not obviously geo or time related, then it is probably a feature column.'
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

    # identify geo column pairs/groups
    # TODO: really only lat/lon should get paired up. if there are multiple lat/lon columns, need llm to pick out which ones are pairs
    remaining_unpaired = [*geo_annotations]
    # geo_pairs = []
    latlon_columns = []
    # geo_type_sets: list[tuple[set[GeoType], list[GeoAnnotation]]] = [
    # ({GeoType.LATITUDE, GeoType.LONGITUDE}, []),
    # ({GeoType.CITY, GeoType.STATE, GeoType.COUNTRY, GeoType.COUNTY}, []),
    # ]
    isolated_geo_columns = []
    for geo in geo_annotations:
        # for groupings, matches in geo_type_sets:
        if geo.geo_type == GeoType.LATITUDE or geo.geo_type == GeoType.LONGITUDE:
            latlon_columns.append(geo)
        else:
            isolated_geo_columns.append(geo)

    pdb.set_trace()

    # while len(remaining_unpaired) > 0:
    #     cur = remaining_unpaired.pop()
    #     pdb.set_trace()
    #     candidates = []  # based on the identified geo type's that could be paired together
    #     identify_column_groups(
    #         agent, df, cur.name, meta,
    #         [i.name for i in remaining_unpaired],
    #         '''\
    #         '''
    #     )
    #     cur.name

    # identify primary geo

    # handling latlon vs lonlat in single coordinate column

    # identify date column pairs/groups
    date_columns = []
    isolated_date_columns = []
    for date in date_annotations:
        if date.date_type == DateType.YEAR or date.date_type == DateType.MONTH or date.date_type == DateType.DAY:
            date_columns.append(date)
        else:
            isolated_date_columns.append(date)

    # identify primary date

    pdb.set_trace()

    return AnnotationSchema(
        geo=geo_annotations,
        date=date_annotations,
        feature=feature_annotations
    )
