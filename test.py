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
    meta = [meta[2]]  # debug, look just at one file

    # shorten the description if necessary
    agent = Agent(model='gpt-4-turbo-preview', timeout=10.0)

    for m in meta:
        if len(m.description) > 1000:
            m.description = shorten_description(m, agent)

        if m.path.suffix == '.csv':
            handle_csv(m, agent)
        elif m.path.suffix == '.xlsx':
            handle_xlsx(m, agent)
        elif m.path.suffix == '.nc':
            handle_netcdf(m, agent)
        elif m.path.suffix == '.tif' or m.path.suffix == '.tiff':
            handle_geotiff(m, agent)
        else:
            raise ValueError(f'Unhandled file type: {m.path.suffix}')


def shorten_description(meta: Meta, agent: Agent) -> str:
    desc = agent.oneshot_sync('You are a helpful assistant.', f'''\
I have a dataset called "{meta.name}" With the following description:
"{meta.description}"
If there is a lot of superfluous information, could you pare it down to just the key details? Output only the new description without any other comments. If there are not superfluous details, output only the original unmodified description.\
''')
    return desc


if __name__ == '__main__':
    main()
