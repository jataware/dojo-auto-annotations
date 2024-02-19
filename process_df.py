from __future__ import annotations

from agent import Message, Role, Agent
from meta import Meta
import pandas as pd
from typing import TypeVar
from utils import enum_to_keys, ask_user, inplace_replace, is_valid_strftime_format

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

    # determine the type of date column for each
    date_type_map = {}
    for col in column_type_map['DATE']:
        date_type = identify_column_type(
            agent, df, col, meta,
            # give the LLM an option for time-like columns, which we will treat as DATE
            enum_to_keys(DateType) + ['TIME'],
            '''\
The column has been identified as containing date/time information.
I need to identify the type of date/time information it contains.\
            '''
        )
        if date_type == 'TIME':
            date_type = 'DATE'  # metadata currently treats times as just DATE
        print(f'LLM identified DATE column "{col}" as a {date_type}')
        date_type_map[col] = date_type

    # identifying the type of geo column for each
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

    # identifying the type of feature column for each
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

    # create the annotations for each column
    geo_annotations: list[GeoAnnotation] = []
    date_annotations: list[DateAnnotation] = []
    feature_annotations: list[FeatureAnnotation] = []

    # maps used to keep track of each annotation's index given its column name
    geo_idxs: dict[str, int] = {}
    date_idxs: dict[str, int] = {}
    feature_idxs: dict[str, int] = {}

    for col in column_type_map['DATE']:
        if date_type_map[col] is not None:
            date_idxs[col] = len(date_annotations)
            date_annotations.append(DateAnnotation(
                name=col,
                display_name=None,
                description=None,
                type=ColumnType.DATE.value,
                date_type=DateType[date_type_map[col]].value,
                primary_date=None,
                time_format='todo',  # have the llm figure this part out
                associated_columns=None,
                qualifies=None,
            ))

    for col in column_type_map['GEO']:
        if geo_type_map[col] is not None:
            geo_idxs[col] = len(geo_annotations)
            geo_annotations.append(GeoAnnotation(
                name=col,
                display_name=None,
                description=None,
                type=ColumnType.GEO.value,
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
            feature_idxs[col] = len(feature_annotations)
            feature_annotations.append(FeatureAnnotation(
                name=col,
                display_name=None,
                description='todo feature description',  # need llm to pick this before making the annotation
                type=ColumnType.FEATURE.value,
                feature_type=FeatureType[feature_type_map[col]].value,
                units=None,
                units_description=None,
                qualifies=None,
                qualifierrole=None,
            ))

    # identify geo lat/lon column pairs
    latlon_columns: list[str] = []
    isolated_geo_columns: list[str] = []
    for geo in geo_annotations:
        # for groupings, matches in geo_type_sets:
        if geo.geo_type == GeoType.LATITUDE or geo.geo_type == GeoType.LONGITUDE:
            latlon_columns.append(geo.name)
        else:
            isolated_geo_columns.append(geo.name)

    latlon_pairs: list[tuple[str, str]] = []
    geo_type_match_map = {
        GeoType.LATITUDE: GeoType.LONGITUDE,
        GeoType.LONGITUDE: GeoType.LATITUDE,
    }
    while len(latlon_columns) > 0:
        cur_col = latlon_columns.pop()
        cur = geo_annotations[geo_idxs[cur_col]]
        candidates = [geo_annotations[geo_idxs[i]] for i in latlon_columns]
        candidates = [i for i in candidates if i.geo_type == geo_type_match_map[cur.geo_type]]
        if len(candidates) == 1:
            # TBD: could have the llm check here if this is a valid match. probably not necessary though
            match, = candidates
            latlon_columns.remove(match.name)

            # ensure lat is first in the pair
            latlon_pairs.append((cur.name, match.name))
            continue

        if len(candidates) == 0:
            isolated_geo_columns.append(cur.name)
            continue

        pdb.set_trace()
        # TODO: else ask the llm to pick the best matching pair if any
        raise NotImplementedError('need to ask the llm to pick the best matching lat/lon pair if any')

    # mark the pairs in the geo annotations (revalidate each annotation with the new info)
    for c0_name, c1_name in latlon_pairs:
        c0 = geo_annotations[geo_idxs[c0_name]]
        inplace_replace(
            geo_annotations,
            c0,
            GeoAnnotation(**{
                **c0.model_dump(),
                'is_geo_pair': c1_name
            })
        )
        # TODO: for now, only one column gets the is_geo_pair attribute
        # c1 = geo_annotations[geo_idxs[c1_name]]
        # inplace_replace(
        #     geo_annotations,
        #     c1,
        #     GeoAnnotation(**{
        #         **c1.model_dump(),
        #         'is_geo_pair': c0_name
        #     })
        # )

    # handling latlon vs lonlat in single coordinate column
    for col in geo_annotations:
        if col.geo_type == GeoType.COORDINATES:
            response = agent.oneshot_sync('You are a helpful assistant.', f'''\
I'm looking at a dataset called "{meta.name}".  I have a column called "{col.name}" with the following values (first 5 rows):
{df[col.name].head().to_string()}
The column has been identified as containing geographic information, and has been marked as containing coordinates.
I need to determine if these coordinates are Latitude,Longitude, or Longitude,Latitude. Without any other comments, please output one of the following options: "LATLON" or "LONLAT" or "UNSURE" if you are unsure.
'''
                                          )
            if response == 'UNSURE':
                print(f'LLM was unsure about the coordinate format for column "{col.name}"')
                continue
            if response not in ('LATLON', 'LONLAT'):
                raise ValueError(f'LLM provided invalid coordinate format for column "{col.name}": {response}')
            coord_format = CoordFormat.LATLON if response == 'LATLON' else CoordFormat.LONLAT
            inplace_replace(
                geo_annotations,
                col,
                GeoAnnotation(**{
                    **col.model_dump(),
                    'coord_format': coord_format
                })
            )
            print(f'LLM identified coordinate column "{col.name}" as having format: "{coord_format.name}"')

    # identify the primary geo
    geo_candidates_str = latlon_pairs + isolated_geo_columns

    def mark_as_primary(geo_name: str):
        geo = geo_annotations[geo_idxs[geo_name]]
        inplace_replace(
            geo_annotations,
            geo,
            GeoAnnotation(**{
                **geo.model_dump(),
                'primary_geo': True
            })
        )

    if len(geo_candidates_str) == 1:
        if len(isolated_geo_columns) == 1:
            mark_as_primary(isolated_geo_columns[0])
        else:
            group, = latlon_pairs
            for geo_name in group:
                mark_as_primary(geo_name)

    elif len(geo_candidates_str) > 1:
        primary_col = agent.oneshot_sync('You are a helpful assistant.', f'''\
I'm looking at a dataset called "{meta.name}".  I have the following geo columns:
{', '.join([ f'{i}:{col}' for i, col in enumerate(geo_candidates_str)])} (noting that columns part of a group are listed together)
Without any other comments, please select the index of the most likely primary geo column(s) from the list above, i.e. please output a single integer (0-{len(geo_candidates_str)-1}) with your selection.
'''
                                         )
        try:
            primary_col = int(primary_col)
            if primary_col < 0 or primary_col >= len(geo_candidates_str):
                raise ValueError(
                    f'LLM provided out of range index for primary geo column. {primary_col=} out of {geo_candidates_str=}')
            if primary_col < len(latlon_pairs):
                group_names = latlon_pairs[primary_col]
                for geo_name in group_names:
                    mark_as_primary(geo_name)
            else:
                primary_col -= len(latlon_pairs)
                mark_as_primary(isolated_geo_columns[primary_col])

            print(f'LLM identified {geo_candidates_str[primary_col]} as the primary geo column(s)')
        except Exception as e:
            pdb.set_trace()
            print(e)

    # identify date column pairs/groups
    date_columns: list[str] = []
    isolated_date_columns: list[str] = []
    for date in date_annotations:
        if date.date_type == DateType.YEAR or date.date_type == DateType.MONTH or date.date_type == DateType.DAY:
            date_columns.append(date.name)
        else:
            isolated_date_columns.append(date.name)

    date_groups: list[tuple[str, ...]] = []
    while len(date_columns) > 0:
        cur_name = date_columns.pop()
        cur = date_annotations[date_idxs[cur_name]]
        candidates = [date_annotations[date_idxs[i]] for i in date_columns]
        candidates = [i for i in candidates if i.date_type != cur.date_type]
        # if the date type of each candidate is unique, and there are 1 or 2 of them, group with cur
        if len(candidates) in (1, 2) and len({i.date_type for i in candidates}) == len(candidates):
            date_groups.append((cur.name, *[c.name for c in candidates]))
            for i in candidates:
                date_columns.remove(i.name)
            continue

        if len(candidates) == 0:
            isolated_date_columns.append(cur.name)
            continue

        pdb.set_trace()
        # TODO: else ask the llm to pick the best matching group if any
        raise NotImplementedError('need to ask the llm to pick the best matching DATE group if any')

    # mark the groups in the date annotations (revalidate each annotation with the new info)
    for group in date_groups:
        # for date_name in group:
        date_name = group[0]  # TODO: for now just take the first column as the one marked with the associated columns
        date = date_annotations[date_idxs[date_name]]
        other_names = [i for i in group if i != date]
        others = [date_annotations[date_idxs[i]] for i in other_names]
        inplace_replace(
            date_annotations,
            date,
            DateAnnotation(**{
                **date.model_dump(),
                # Dirty hack: convert DateType to TimeField.
                # For now, we can only identify year, month, day groups. No hour or, minute columns
                'associated_columns': {TimeField[i.date_type.name]: i.name for i in others}
            })
        )

    # identify primary date
    date_candidates_str = date_groups + isolated_date_columns

    def mark_as_primary(date_name: str):
        date = date_annotations[date_idxs[date_name]]
        inplace_replace(
            date_annotations,
            date,
            DateAnnotation(**{
                **date.model_dump(),
                'primary_date': True
            })
        )

    if len(date_candidates_str) == 1:
        if len(isolated_date_columns) == 1:
            mark_as_primary(isolated_date_columns[0])
        else:
            group, = date_groups
            for date_name in group:
                mark_as_primary(date_name)

    elif len(date_candidates_str) > 1:
        primary_col = agent.oneshot_sync('You are a helpful assistant.', f'''\
I'm looking at a dataset called "{meta.name}".  I have the following date columns:
{', '.join([ f'{i}:{col}' for i, col in enumerate(date_candidates_str)])} (noting that columns part of a group are listed together)
Without any other comments, please select the index of the most likely primary date column(s) from the list above, i.e. please output a single integer (0-{len(date_candidates_str)-1}) with your selection. 
''')
        try:
            primary_col = int(primary_col)
            if primary_col < 0 or primary_col >= len(date_candidates_str):
                raise ValueError(
                    f'LLM provided out of range index for primary date column. {primary_col=} out of {date_candidates_str=}')
            if primary_col < len(date_groups):
                group_names = date_groups[primary_col]
                for date_name in group_names:
                    mark_as_primary(date_name)
            else:
                primary_col -= len(date_groups)
                mark_as_primary(isolated_date_columns[primary_col])

            print(f'LLM identified {date_candidates_str[primary_col]} as the primary date column(s)')
        except Exception as e:
            pdb.set_trace()
            print(e)

    # identify the format string of DateType.DATE columns
    for date in date_annotations:
        if date.date_type in (DateType.YEAR, DateType.MONTH, DateType.DAY, DateType.DATE):
            col = date.name
            response = agent.oneshot_sync('You are a helpful assistant.', f'''\
I'm looking at a dataset called "{meta.name}".  I have a column called "{col}" with the following values (first 5 rows):
{df[col].head().to_string()}
The column has been identified as containing date/time information, and has been marked as a {date.date_type.name} column.
I need to identify the strftime format for this field. Without any other comments, please output a valid strftime format string or UNSURE if you are unsure.
'''
                                          )
            if response == 'UNSURE':
                print(f'LLM was unsure about the time format for {date.date_type.name} column "{col}"')
                continue  # TODO: could ask the user here. For now just skip

            # strip any wrapping quotes
            fmt = response.strip('\'"`')

            # TODO: check if format string is a valid format string
            assert is_valid_strftime_format(
                fmt), f'LLM provided invalid strftime format string for {date.date_type.name} column "{col}": {fmt}'

            inplace_replace(
                date_annotations,
                date,
                DateAnnotation(**{
                    **date.model_dump(),
                    'time_format': fmt
                })
            )

            print(f'LLM identified {date.type.name}/{date.date_type.name} column "{col}" strftime format: "{fmt}"')

    pdb.set_trace()
    return AnnotationSchema(
        geo=geo_annotations,
        date=date_annotations,
        feature=feature_annotations
    )
