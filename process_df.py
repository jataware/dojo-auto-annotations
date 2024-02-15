from __future__ import annotations

from agent import Message, Role, Agent
from meta import Meta
import pandas as pd
from typing import TypeVar
from utils import enum_to_keys, ask_user, inplace_replace

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


# def identify_column_groups(agent: Agent, df: pd.DataFrame, col: str, meta: Meta, options: list[T], prompt: str) -> T | None:
#     options_or_unsure = options + ['UNSURE']
#     options_or_none = options + ['NONE']

#     # initial messages to the llm
#     messages = [
#         Message(Role.system, 'You are a helpful assistant.'),
#         Message(Role.user, f'''\
# I have a dataset called "{meta.name}" with the following description:
# "{meta.description}"
# I have a column called "{col}" with the following values (first 5 rows):
# {df[col].head().to_string()}
# {prompt}
# Please select zero or more of the following options: {', '.join(options)}. If you are unsure, only output UNSURE. Write your answer without any other comments.\
# '''
#                 )
#     ]

#     # ask the model to identify the type of the column
#     res = agent.multishot_sync(messages)

#     pdb.set_trace()
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


    # identify geo lat/lon column pairs
    latlon_columns: list[GeoAnnotation] = []
    isolated_geo_columns = []
    for geo in geo_annotations:
        # for groupings, matches in geo_type_sets:
        if geo.geo_type == GeoType.LATITUDE or geo.geo_type == GeoType.LONGITUDE:
            latlon_columns.append(geo)
        else:
            isolated_geo_columns.append(geo)


    latlon_pairs: list[tuple[GeoAnnotation, GeoAnnotation]] = []
    geo_type_match_map = {
        GeoType.LATITUDE: GeoType.LONGITUDE,
        GeoType.LONGITUDE: GeoType.LATITUDE,
    }
    while len(latlon_columns) > 0:
        cur = latlon_columns.pop()
        candidates = [i for i in latlon_columns if i.geo_type == geo_type_match_map[cur.geo_type]]
        if len(candidates) == 1:
            #TBD: could have the llm check here if this is a valid match. probably not necessary though
            match, = candidates
            latlon_columns.remove(match)

            # ensure lat is first in the pair
            latlon_pairs.append((cur, match))
            continue

        if len(candidates) == 0:
            isolated_geo_columns.append(cur)
            continue

        pdb.set_trace()
        #TODO: else ask the llm to pick the best matching pair if any
        raise NotImplementedError('need to ask the llm to pick the best matching lat/lon pair if any')

    # mark the pairs in the geo annotations (revalidate each annotation with the new info)
    for c0, c1 in latlon_pairs:
        inplace_replace(
            geo_annotations,
            c0,
            GeoAnnotation(**{
                **c0.model_dump(),
                'is_geo_pair': c1.name
            })
        )
        #TODO: for now, only one column gets the is_geo_pair attribute
        # inplace_replace(
        #     geo_annotations,
        #     c1,
        #     GeoAnnotation(**{
        #         **c1.model_dump(),
        #         'is_geo_pair': c0.name
        #     })
        # )



    # handling latlon vs lonlat in single coordinate column
    # TODO:...

    # identify date column pairs/groups
    date_columns: list[DateAnnotation] = []
    isolated_date_columns: list[DateAnnotation] = []
    for date in date_annotations:
        if date.date_type == DateType.YEAR or date.date_type == DateType.MONTH or date.date_type == DateType.DAY:
            date_columns.append(date)
        else:
            isolated_date_columns.append(date)


    date_groups: list[tuple[DateAnnotation,...]] = []
    while len(date_columns) > 0:
        cur = date_columns.pop()
        candidates = [i for i in date_columns if i.date_type != cur.date_type]
        # if the date type of each candidate is unique, and there are 1 or 2 of them, group with cur
        if len(candidates) in (1, 2) and len({i.date_type for i in candidates}) == len(candidates):
            date_groups.append((cur, *candidates))
            for i in candidates:
                date_columns.remove(i)
            continue
        
        if len(candidates) == 0:
            isolated_date_columns.append(cur)
            continue

        pdb.set_trace()
        # TODO: else ask the llm to pick the best matching group if any
        raise NotImplementedError('need to ask the llm to pick the best matching DATE group if any')
    
    # mark the groups in the date annotations (revalidate each annotation with the new info)
    for i, group in enumerate(date_groups):
        # for date in group:
            date = group[0] #TODO: for now just take the first column as the one marked with the associated columns
            others = [i for i in group if i != date]
            inplace_replace(
                date_annotations,
                date,
                new_date:=DateAnnotation(**{
                    **date.model_dump(),
                    # Dirty hack: convert DateType to TimeField. 
                    # For now, we can only identify year, month, day groups. No hour or, minute columns
                    'associated_columns': {TimeField[i.date_type.name]: i.name for i in others}
                })
            )
            #have to update the group with the new date
            date_groups[i] = (new_date, *others)
    

    # identify primary date
    group_candidates_str = [tuple(date.name for date in group) for group in date_groups]
    isolated_candidates_str = [date.name for date in isolated_date_columns]
    candidates_str = group_candidates_str + isolated_candidates_str
    if len(candidates_str) == 1:
        if len(isolated_date_columns) == 1:
            inplace_replace(
                date_annotations,
                isolated_date_columns[0],
                DateAnnotation(**{
                    **isolated_date_columns[0].model_dump(),
                    'primary_date': True
                })
            )
        else:
            group, = date_groups
            for date in group:
                inplace_replace(
                    date_annotations,
                    date,
                    DateAnnotation(**{
                        **date.model_dump(),
                        'primary_date': True
                    })
                )

    elif len(candidates_str) > 1:
        primary_col = agent.oneshot_sync('You are a helpful assistant.', f'''\
I'm looking at a dataset called "{meta.name}".  I have the following date columns:
{', '.join([ f'{i}:{col}' for i, col in enumerate(candidates_str)])} (noting that columns part of a group are listed together)
Without any other comments, please select the index of the most likely primary date column(s) from the list above, i.e. please output a single integer with your selection. 
''')
        try:
            primary_col = int(primary_col)
            if primary_col < 0 or primary_col >= len(candidates_str):
                raise ValueError(f'LLM provided out of range index for primary date column. {primary_col=} out of {candidates_str=}')
            if primary_col < len(group_candidates_str):
                group = date_groups[primary_col]
                for date in group:
                    inplace_replace(
                        date_annotations,
                        date,
                        DateAnnotation(**{
                            **date.model_dump(),
                            'primary_date': True
                        })
                    )
            else:
                primary_col -= len(group_candidates_str)
                inplace_replace(
                    date_annotations,
                    isolated_date_columns[primary_col],
                    DateAnnotation(**{
                        **isolated_date_columns[primary_col].model_dump(),
                        'primary_date': True
                    })
                )
            print(f'LLM identified {candidates_str[primary_col]} as the primary date column(s)')
        except Exception as e:
            pdb.set_trace()
            print(e)

    # identify the format string of DateType.DATE columns
    for date in date_annotations:
        if date.date_type in (DateType.YEAR, DateType.MONTH, DateType.DAY, DateType.DATE):
            col = date.name
            fmt = agent.oneshot_sync('You are a helpful assistant.', f'''\
I'm looking at a dataset called "{meta.name}".  I have a column called "{col}" with the following values (first 5 rows):
{df[col].head().to_string()}
The column has been identified as containing date/time information, and has been marked as a {date.date_type.name} column.
I need to identify the strftime format for this field. Without any other comments, please output a valid strftime format string or UNSURE if you are unsure.
'''
            )
            if fmt == 'UNSURE':
                print(f'LLM was unsure about the time format for {date.date_type.name} column "{col}"')
                continue #TODO: could ask the user here. For now just skip

            inplace_replace(
                date_annotations,
                date,
                DateAnnotation(**{
                    **date.model_dump(),
                    'time_format': fmt
                })
            )

    pdb.set_trace()
    return AnnotationSchema(
        geo=geo_annotations,
        date=date_annotations,
        feature=feature_annotations
    )
