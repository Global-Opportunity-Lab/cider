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

import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import dask.dataframe as dd
import geopandas as gpd  # type: ignore[import]
import pandas as pd
import pyarrow.parquet as pq
from box import Box
from geopandas import GeoDataFrame
from pandas import DataFrame as PandasDataFrame

from helpers.utils import filter_dates_dataframe, get_dask_client

# Type alias for Dask DataFrame
DaskDataFrame = dd.DataFrame


def _validate_parquet_dir(parquet_dir: Path) -> None:
    """
    Check that every .parquet file in the directory can be opened as Parquet.
    Raises with the path and cause if any file is invalid (so you know which file to fix).
    """
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        return
    for p in files:
        try:
            pq.ParquetFile(p)
        except Exception as e:
            raise ValueError(
                f"Invalid or corrupted Parquet file: {p}\n"
                f"Reason: {e}\n"
                f"Check that the file was written completely (e.g. no truncated write) and is not empty."
            ) from e


class IOUtils:
    
    def __init__(
        self,
        cfg: Box, 
        data_format: Box
    ):
        self.cfg = cfg
        self.data_format = data_format
        self.client = get_dask_client(cfg)

    def _get_date_filter_for_dataset(self, dataset_name: str) -> Optional[tuple]:
        """If config has params.featurization.start_date and end_date, return (start, end, file_timestamp_col)."""
        featurization = getattr(self.cfg.get('params', Box()), 'featurization', Box())
        start = getattr(featurization, 'start_date', None)
        end = getattr(featurization, 'end_date', None)
        if start is None or end is None:
            return None
        col_names = getattr(self.cfg.col_names, dataset_name, Box())
        timestamp_col_file = getattr(col_names, 'timestamp', 'timestamp')
        return (str(start), str(end), str(timestamp_col_file))

    
    def load_generic(
        self,
        fpath: Optional[Path] = None,
        df: Optional[Union[DaskDataFrame, PandasDataFrame]] = None,
        filter_start_date: Optional[str] = None,
        filter_end_date: Optional[str] = None,
        filter_timestamp_column: Optional[str] = None,
    ) -> DaskDataFrame:
        """
        Load data from file or dataframe.

        When filter_start_date, filter_end_date, and filter_timestamp_column are set and the
        source is Parquet, only rows with timestamp in [start, end) are read (filter-on-read).
        CSV is not filtered on read; date filter is applied later in load_cdr / load_recharges / etc.
        """
        # Prefer provided dataframe over reading from disk
        if df is not None:
            if isinstance(df, PandasDataFrame):
                return dd.from_pandas(df)
            if isinstance(df, dd.DataFrame):
                return df
            raise TypeError("df must be a pandas or dask dataframe")

        use_date_filter = (
            filter_start_date is not None
            and filter_end_date is not None
            and filter_timestamp_column is not None
        )
        # Do not pass filters= to read_parquet: Dask serializes the filter expression to
        # workers and the scalar type is lost (timestamp becomes string), causing
        # "Function 'equal' has no kernel matching input types (timestamp[us], string)".
        # We filter immediately after read instead (see below).

        # Load from file
        if fpath is not None:
            fpath = Path(fpath) if not isinstance(fpath, Path) else fpath
            # Load data if in a single file
            if fpath.is_file():

                if fpath.suffix == '.csv':
                    df = dd.read_csv(
                        str(fpath),
                        dtype={
                            'caller_id': 'str', 'recipient_id': 'str', 'name': 'str', 'duration': 'float64',
                            'cell_id': 'str', 'lac_id': 'str', 'antenna_id': 'str', 'tower_id': 'str',
                        },
                        assume_missing=True,
                    )

                elif fpath.suffix == '.parquet':
                    try:
                        pq.ParquetFile(fpath)
                    except Exception as e:
                        raise ValueError(
                            f"Invalid or corrupted Parquet file: {fpath}\n"
                            f"Reason: {e}\n"
                            f"Check that the file was written completely and is not empty."
                        ) from e
                    df = dd.read_parquet(str(fpath))

                else:
                    raise ValueError(f'File with unknown extension {fpath}.')

            # Load data if in chunks
            else:
                try:
                    example_file = next(fpath.iterdir())

                except StopIteration:
                    raise ValueError(f'Directory {fpath} is empty.')

                except FileNotFoundError as e:
                    raise ValueError(f"Could not locate or read data for '{fpath}'")

                if example_file.suffix == '.csv':
                    df = dd.read_csv(
                        str(fpath / '*.csv'),
                        dtype={
                            'caller_id': 'str', 'recipient_id': 'str', 'name': 'str', 'duration': 'float64',
                            'cell_id': 'str', 'lac_id': 'str', 'antenna_id': 'str', 'tower_id': 'str',
                        },
                        assume_missing=True,
                    )

                elif example_file.suffix == '.parquet':
                    _validate_parquet_dir(fpath)
                    df = dd.read_parquet(str(fpath / '*.parquet'))

                else:
                    raise ValueError(f'Found file with unknown extension {example_file}.')

            if use_date_filter and filter_timestamp_column in df.columns:
                df = filter_dates_dataframe(
                    df, filter_start_date, filter_end_date, colname=filter_timestamp_column
                )

            return df

        # Issue with filename/dataframe provided
        raise ValueError('No filename or pandas/dask dataframe provided.')


    def check_cols(
        self,
        df: Union[GeoDataFrame, PandasDataFrame, DaskDataFrame],
        dataset_name: str,
    ) -> None:
        """
        Check that the df has all required columns

        Args:
            df: dataframe
            dataset_name: name of dataset, to be used in error messages.
        """
        dataset_data_format = self.data_format[dataset_name]
        required_cols = set(dataset_data_format.required)

        columns_present = set(df.columns)

        if not required_cols.issubset(columns_present):
            raise ValueError(
                f"{dataset_name} data format incorrect. {dataset_name} must include the following columns: {', '.join(required_cols)}, "
                f"instead found {', '.join(columns_present)}"
            )


    def check_colvalues(
        self, df: DaskDataFrame, colname: str, colvalues: list, error_msg: str
    ) -> None:
        """
        Check that a column has all required values

        Args:
            df: dask df
            colname: column to check
            colvalues: required values
            error_msg: error message to print if values don't match
        """
        unique_values = set(df[colname].unique().compute().tolist())
        if not unique_values.issubset(set(colvalues)):
            raise ValueError(error_msg)


    def standardize_col_names(
        self, df: DaskDataFrame, col_names: Dict[str, str]
    ) -> DaskDataFrame:
        """
        Rename columns, as specified in config file, to standard format

        Args:
            df: dask df
            col_names: mapping between standard column names and existing ones

        Returns: 
            Dask df with standardized column names
        """
        col_mapping = {v: k for k, v in col_names.items()}

        # Only rename columns that exist in the dataframe
        rename_dict = {col: col_mapping[col] for col in df.columns if col in col_mapping}
        
        if rename_dict:
            df = df.rename(columns=rename_dict)

        return df

    def load_dataset(
        self,
        dataset_name: str,
        fpath: Optional[Path] = None,
        provided_df: Optional[Union[DaskDataFrame, PandasDataFrame]] = None,
        filter_start_date: Optional[str] = None,
        filter_end_date: Optional[str] = None,
        filter_timestamp_col_file: Optional[str] = None,
    ) -> DaskDataFrame:
        """
        Load a dataset with standardized column names and validation.

        When filter_* are set, Parquet is read with row subset on read; CSV is filtered after load.
        """
        dataset = self.load_generic(
            fpath,
            provided_df,
            filter_start_date=filter_start_date,
            filter_end_date=filter_end_date,
            filter_timestamp_column=filter_timestamp_col_file,
        )

        if dataset_name in self.cfg.col_names:
            dataset = self.standardize_col_names(dataset, self.cfg.col_names[dataset_name])

        self.check_cols(dataset, dataset_name)

        # Date filter is applied in load_cdr / load_recharges / etc. after timestamp is cleaned,
        # so we don't filter here on raw timestamp (which can yield 0 rows if format differs).
        return dataset    


    def load_cdr(
        self,
        fpath: Optional[Path] = None,
        df: Optional[Union[DaskDataFrame, PandasDataFrame]] = None
    ) -> DaskDataFrame:
        """
        Load CDR data into dask df

        Returns: 
            Dask df
        """
        date_filter = self._get_date_filter_for_dataset('cdr')
        cdr = self.load_dataset(
            dataset_name='cdr',
            fpath=fpath,
            provided_df=df,
            filter_start_date=date_filter[0] if date_filter else None,
            filter_end_date=date_filter[1] if date_filter else None,
            filter_timestamp_col_file=date_filter[2] if date_filter else None,
        )

        # Check txn_type column
        error_msg = 'CDR format incorrect. Column txn_type can only include call and text.'
        self.check_colvalues(cdr, 'txn_type', ['call', 'text'], error_msg)

        # Normalize international column to CIDER's expected values (domestic, international, other)
        # so that 0/1, intl/national, or other encodings are accepted.
        default_international_map = {
            '0': 'domestic', 0: 'domestic', '0.0': 'domestic',
            '1': 'international', 1: 'international', '1.0': 'international',
            'domestic': 'domestic', 'international': 'international', 'other': 'other',
            'national': 'domestic', 'intl': 'international',
        }
        international_map = getattr(
            getattr(self.cfg.get('params', Box()), 'cdr', Box()),
            'international_map',
            None,
        )
        if isinstance(international_map, dict):
            default_international_map = {**default_international_map, **international_map}
        cdr['international'] = cdr['international'].astype(str).str.strip().str.lower().replace(default_international_map)
        # Map any remaining unknown values to 'other' so downstream code does not break
        allowed = {'domestic', 'international', 'other'}
        cdr['international'] = cdr['international'].where(
            cdr['international'].isin(allowed),
            'other',
        )

        # Check international column has only allowed values (after normalization)
        error_msg = 'CDR format incorrect. Column international can only include domestic, international, and other (or values mappable via params.cdr.international_map).'
        self.check_colvalues(cdr, 'international', ['domestic', 'international', 'other'], error_msg)

        # if no recipient antennas are present, add a null column to enable the featurizer to work
        if 'recipient_antenna' not in cdr.columns:
            cdr['recipient_antenna'] = None
            cdr['recipient_antenna'] = cdr['recipient_antenna'].astype('object')

        # Clean timestamp column
        cdr = self.clean_timestamp_and_add_day_column(cdr, 'timestamp')

        # Subset by config date range (after timestamp is datetime so comparison is correct)
        if date_filter is not None:
            cdr = filter_dates_dataframe(cdr, date_filter[0], date_filter[1], colname='timestamp')

        # Clean duration column
        cdr['duration'] = cdr['duration'].astype('float')

        return cdr


    def load_labels(
        self,
        fpath: Path = None
    ) -> DaskDataFrame:
        """
        Load labels on which to train ML model.
        """
        labels = self.load_dataset('labels', fpath=fpath)

        if 'weight' not in labels.columns:
            labels['weight'] = 1

        return labels[['name', 'label', 'weight']]


    def load_antennas(
        self,
        fpath: Optional[Path] = None,
        df: Optional[Union[DaskDataFrame, PandasDataFrame]] = None
    ) -> DaskDataFrame:
        """
        Load antennas' dataset, and print % of antennas that are missing coordinates

        Returns: 
            Dask df
        """
        antennas = self.load_dataset('antennas', fpath=fpath, provided_df=df)

        antennas['latitude'] = antennas['latitude'].astype('float')
        antennas['longitude'] = antennas['longitude'].astype('float')
        
        total_count = len(antennas)
        valid_count = len(antennas[['latitude', 'longitude']].dropna())
        number_missing_location = total_count - valid_count
        
        if number_missing_location > 0:
            print(f'Warning: {number_missing_location} antennas missing location')

        return antennas


    def load_recharges(
        self,
        fpath: Optional[Path] = None,
        df: Optional[Union[DaskDataFrame, PandasDataFrame]] = None
    ) -> DaskDataFrame:
        """
        Load recharges dataset

        Returns: 
            Dask df
        """
        date_filter = self._get_date_filter_for_dataset('recharges')
        recharges = self.load_dataset(
            'recharges',
            fpath=fpath,
            provided_df=df,
            filter_start_date=date_filter[0] if date_filter else None,
            filter_end_date=date_filter[1] if date_filter else None,
            filter_timestamp_col_file=date_filter[2] if date_filter else None,
        )

        # Clean timestamp column
        recharges = self.clean_timestamp_and_add_day_column(recharges, 'timestamp')

        if date_filter is not None:
            recharges = filter_dates_dataframe(recharges, date_filter[0], date_filter[1], colname='timestamp')

        # Clean amount column
        recharges['amount'] = recharges['amount'].astype('float')

        return recharges


    def load_mobiledata(
        self,
        fpath: Optional[Path] = None,
        df: Optional[Union[DaskDataFrame, PandasDataFrame]] = None
    ) -> DaskDataFrame:
        """
        Load mobile data dataset
        """
        date_filter = self._get_date_filter_for_dataset('mobiledata')
        mobiledata = self.load_dataset(
            'mobiledata',
            fpath=fpath,
            provided_df=df,
            filter_start_date=date_filter[0] if date_filter else None,
            filter_end_date=date_filter[1] if date_filter else None,
            filter_timestamp_col_file=date_filter[2] if date_filter else None,
        )

        # Clean timestamp column
        mobiledata = self.clean_timestamp_and_add_day_column(mobiledata, 'timestamp')

        if date_filter is not None:
            mobiledata = filter_dates_dataframe(mobiledata, date_filter[0], date_filter[1], colname='timestamp')

        # Clean volume column
        mobiledata['volume'] = mobiledata['volume'].astype('float')

        return mobiledata


    def load_mobilemoney(
        self,
        fpath: Optional[Path] = None,
        df: Optional[Union[DaskDataFrame, PandasDataFrame]] = None,
        verify: bool = True
    ) -> DaskDataFrame:
        """
        Load mobile money dataset

        Returns: 
            Dask df
        """
        date_filter = self._get_date_filter_for_dataset('mobilemoney')
        mobilemoney = self.load_dataset(
            'mobilemoney',
            fpath=fpath,
            provided_df=df,
            filter_start_date=date_filter[0] if date_filter else None,
            filter_end_date=date_filter[1] if date_filter else None,
            filter_timestamp_col_file=date_filter[2] if date_filter else None,
        )

        # Normalize txn_type to CIDER's expected values so different encodings are accepted
        txn_types = ['cashin', 'cashout', 'p2p', 'billpay', 'other']
        default_txn_type_map = {
            'cashin': 'cashin', 'cash out': 'cashout', 'cashout': 'cashout',
            'cash in': 'cashin', 'deposit': 'cashin', 'withdrawal': 'cashout',
            'send': 'p2p', 'receive': 'p2p', 'p2p': 'p2p', 'transfer': 'p2p',
            'billpay': 'billpay', 'bill pay': 'billpay', 'bill': 'billpay',
            'other': 'other',
        }
        txn_type_map = getattr(
            getattr(self.cfg.get('params', Box()), 'mobilemoney', Box()),
            'txn_type_map',
            None,
        )
        if isinstance(txn_type_map, dict):
            default_txn_type_map = {**default_txn_type_map, **{str(k).strip().lower(): v for k, v in txn_type_map.items()}}
        mobilemoney['txn_type'] = mobilemoney['txn_type'].astype(str).str.strip().str.lower().replace(default_txn_type_map)
        mobilemoney['txn_type'] = mobilemoney['txn_type'].where(
            mobilemoney['txn_type'].isin(txn_types),
            'other',
        )
        error_msg = 'Mobile money format incorrect. Column txn_type can only include ' + ', '.join(txn_types) + ' (or values mappable via params.mobilemoney.txn_type_map).'
        self.check_colvalues(mobilemoney, 'txn_type', txn_types, error_msg)

        # Clean timestamp column
        mobilemoney = self.clean_timestamp_and_add_day_column(mobilemoney, 'timestamp')

        if date_filter is not None:
            mobilemoney = filter_dates_dataframe(mobilemoney, date_filter[0], date_filter[1], colname='timestamp')

        # Clean amount column
        mobilemoney['amount'] = mobilemoney['amount'].astype('float')

        # Clean balance columns
        for c in mobilemoney.columns:
            if 'balance' in c:
                mobilemoney[c] = mobilemoney[c].astype('float')

        return mobilemoney


    def load_shapefile(self, fpath: Path) -> GeoDataFrame:
        """
        Load shapefile and make sure it has the right columns

        Args:
            fpath: path to file, which can be .shp or .geojson

        Returns: 
            GeoDataFrame
        """
        shapefile = gpd.read_file(fpath)

        # Verify that columns are correct
        self.check_cols(shapefile, 'shapefile')

        # Verify that the geometry column has been loaded correctly
        assert shapefile.dtypes['geometry'] == 'geometry'

        shapefile['region'] = shapefile['region'].astype(str)

        return shapefile


    def clean_timestamp_and_add_day_column(
        self,
        df: DaskDataFrame,
        existing_timestamp_column_name: str
    ) -> DaskDataFrame:
        """
        Convert timestamp column to datetime and add day column
        
        Args:
            df: Dask DataFrame
            existing_timestamp_column_name: Name of timestamp column
            
        Returns:
            Dask DataFrame with cleaned timestamp and day columns
        """
        # Avoid head(1) on empty dataframe (prevents Dask "Insufficient elements for head" warning)
        nrows = df.shape[0].compute()
        if nrows == 0:
            raise ValueError(
                'Cannot clean timestamp: dataframe has no rows. '
                'If you set params.featurization.start_date and end_date, the date filter may have excluded all data. '
                'Check that your data has timestamps in that range, or remove the date range from config.'
            )
        # Sample one row for timestamp format (use npartitions=-1 so we get a row even if
        # the first partition is empty after date filter).
        sample = df[existing_timestamp_column_name].head(1, npartitions=-1)
        if len(sample) == 0:
            raise ValueError(
                'Cannot clean timestamp: no sample row (date filter may have left only empty partitions). '
                'Check params.featurization.start_date and end_date.'
            )
        existing_timestamp_sample = sample.iloc[0]
        timestamp_with_time_regex = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"

        has_time_info = bool(re.match(timestamp_with_time_regex, str(existing_timestamp_sample)))

        timestamp_format = (
            '%Y-%m-%d %H:%M:%S' if has_time_info else '%Y-%m-%d'
        )

        # Convert to datetime
        df['timestamp'] = dd.to_datetime(df[existing_timestamp_column_name], format=timestamp_format)
        
        # Add day column (truncated to day)
        df['day'] = df['timestamp'].dt.floor('D')

        return df


    def load_phone_numbers_to_featurize(
        self,
        fpath: Optional[Path] = None,
        df: Optional[Union[DaskDataFrame, PandasDataFrame]] = None,
    ) -> DaskDataFrame:
        """
        Load list of phone numbers to featurize
        
        Args:
            fpath: Path to file with phone numbers
            df: Existing dataframe with phone numbers
            
        Returns:
            Dask DataFrame with phone_number column
        """
        phone_numbers_to_featurize = self.load_dataset(
            'phone_numbers_to_featurize', fpath=fpath, provided_df=df
        )
        phone_numbers_to_featurize = phone_numbers_to_featurize[['phone_number']]

        distinct_count = phone_numbers_to_featurize['phone_number'].nunique().compute()
        length = len(phone_numbers_to_featurize)

        if distinct_count != length:
            n_duplicates = int(length) - int(distinct_count)
            phone_numbers_to_featurize = phone_numbers_to_featurize.drop_duplicates(subset=['phone_number'])
            warnings.warn(
                f'Dropped {n_duplicates} duplicate phone number(s) from the list to featurize: '
                f'list had {length} rows and {distinct_count} distinct values. '
                f'Featurization will run on {distinct_count} unique numbers.',
                UserWarning,
                stacklevel=2,
            )

        return phone_numbers_to_featurize
