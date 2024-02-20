from __future__ import annotations

from agent import Agent, set_openai_key
from meta import Meta, get_meta
from process_df import handle_csv, handle_xlsx
from process_xr import handle_netcdf, handle_geotiff

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
    meta = [*meta[:12]]  # debug, look just at the csv/xlsx files

    # shorten the description if necessary
    agent = Agent(model='gpt-4-turbo-preview', timeout=10.0)

    for m in meta:
        print(m)
        if len(m.description) > 1000:
            m.description = shorten_description(m, agent)

        if m.path.suffix == '.csv':
            annotations = handle_csv(m, agent)
        elif m.path.suffix == '.xlsx':
            annotations = handle_xlsx(m, agent)
        elif m.path.suffix == '.nc':
            annotations = handle_netcdf(m, agent)
        elif m.path.suffix == '.tif' or m.path.suffix == '.tiff':
            annotations = handle_geotiff(m, agent)
        else:
            raise ValueError(f'Unhandled file type: {m.path.suffix}')

        print(annotations)
        print('\n\n')


def shorten_description(meta: Meta, agent: Agent) -> str:
    desc = agent.oneshot_sync('You are a helpful assistant.', f'''\
I have a dataset called "{meta.name}" With the following description:
"""
{meta.description}
"""
I would like to ensure that it is just a simple description purely about the data without any other superfluous information. Things to remove include contact info, bibliographies, URLs, etc. If there is a lot of superfluous information, could you pare it down to just the key details? Output only the new description without any other comments. If there are not superfluous details, output only the original unmodified description.\
''')
    return desc


def main2():
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser()
    parser.add_argument('--path', action='store', type=Path)
    parser.add_argument('--name', action='store', type=str)
    parser.add_argument('--description', action='store', type=str)
    args = parser.parse_args()

    meta = Meta(args.path, args.name, args.description)

    set_openai_key()

    agent = Agent(model='gpt-4-turbo-preview', timeout=10.0)

    if args.path.suffix == '.csv':
        annotations = handle_csv(meta, agent)
    elif args.path.suffix == '.xlsx':
        annotations = handle_xlsx(meta, agent)
    elif args.path.suffix == '.nc':
        annotations = handle_netcdf(meta, agent)
    elif args.path.suffix == '.tif' or args.path.suffix == '.tiff':
        annotations = handle_geotiff(meta, agent)
    else:
        raise ValueError(f'Unhandled file type: {args.path.suffix}')

    print(annotations)


if __name__ == '__main__':
    # main()
    main2()
