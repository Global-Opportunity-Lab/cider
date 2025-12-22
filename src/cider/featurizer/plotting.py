# Copyright ©2022-2023. The Regents of the University of California
# (Regents). All Rights Reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the
# distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import pandas as pd


def plot_timeseries_diagnostics(
    timeseries_diagnostics_df: pd.DataFrame,
    value_column: str,
    groupby_column: str | None = None,
    plot_title: str | None = None,
    **plotting_kwargs,
):
    """
    Plot timeseries diagnostics.

    Args:
        timeseries_diagnostics_df: Dataframe with timeseries diagnostics
        value_column: Column name in timeseries_diagnostics_df to plot as y-axis
        groupby_column: Column name in timeseries_diagnostics_df to group by for multiple lines
        plot_title: Title for the plot. If None, a default title will be used
        **plotting_kwargs: Additional keyword arguments for pandas plot method (e.g. color, cmap, etc.)
    """
    if groupby_column and groupby_column not in timeseries_diagnostics_df.columns:
        raise ValueError(
            f"Timeseries diagnostics df must contain `{groupby_column}` column to group by"
        )
    if value_column not in timeseries_diagnostics_df.columns:
        raise ValueError(
            f"Timeseries diagnostics df must contain `{value_column}` column to plot"
        )
    if "day" not in timeseries_diagnostics_df.columns:
        raise ValueError(
            "Timeseries diagnostics df must contain `day` column for x-axis"
        )

    with mpl.rc_context(fname=Path(__file__).parent / "../matplotlibrc"):
        fig, ax = plt.subplots(figsize=(10, 6))

        if groupby_column:
            grouped_df = timeseries_diagnostics_df.groupby(groupby_column)
        else:
            grouped_df = timeseries_diagnostics_df
        grouped_df[value_column].plot(ax=ax, **plotting_kwargs)
        ax.set_title(plot_title if plot_title else f"{value_column} Over Time")
        ax.set_xlabel("Date")
        ax.set_xticklabels(timeseries_diagnostics_df["day"], rotation=45)
        ax.set_ylabel(value_column)
        ax.legend(title=groupby_column if groupby_column else None)

        fig.tight_layout()
    return fig
