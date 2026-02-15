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
from pathlib import Path
from typing import Dict, List, Optional, Union
import re

import dask.dataframe as dd
import geopandas as gpd  # type: ignore[import]
import pandas as pd
from box import Box
from geopandas import GeoDataFrame
from pandas import DataFrame as PandasDataFrame

from helpers.utils import get_dask_client

# Type alias for Dask DataFrame
DaskDataFrame = dd.DataFrame


class IOUtils:
    
    def __init__(
        self,
        cfg: Box, 
        data_format: Box
    ):
        self.cfg = cfg
        self.data_format = data_format
        self.client = get_dask_client(cfg)

    
    def load_generic(
        self,
        fpath: Optional[Path] = None,
        df: Optional[Union[DaskDataFrame, PandasDataFrame]] = None
    ) -> DaskDataFrame:
        """
        Load data from file or dataframe
        
        Args:
            fpath: Path to data file or directory
            df: Existing dataframe (Dask or Pandas)
            
        Returns:
            Dask DataFrame
        """
        # Load from file
        if fpath is not None:
            # Load data if in a single file
            if fpath.is_file():

                if fpath.suffix == '.csv':
                    df = dd.read_csv(str(fpath), dtype={'caller_id': 'str', 'recipient_id': 'str', 'name': 'str'})

                elif fpath.suffix == '.parquet':
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
                    df = dd.read_csv(str(fpath / '*.csv'), dtype={'caller_id': 'str', 'recipient_id': 'str', 'name': 'str'})

                elif example_file.suffix == '.parquet':
                    df = dd.read_parquet(str(fpath / '*.parquet'))

                else:
                    raise ValueError(f'Found file with unknown extension {example_file}.')

        # Load from pandas dataframe
        elif df is not None:
            if isinstance(df, PandasDataFrame):
                df = dd.from_pandas(df, npartitions=4)
            # If already Dask, return as-is

        # Issue with filename/dataframe provided
        else:
            raise ValueError('No filename or pandas/dask dataframe provided.')

        return df


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
        provided_df: Optional[Union[DaskDataFrame, PandasDataFrame]] = None
    ) -> DaskDataFrame:
        """
        Load a dataset with standardized column names and validation
        
        Args:
            dataset_name: Name of the dataset (for validation)
            fpath: Path to data file
            provided_df: Existing dataframe
            
        Returns:
            Dask DataFrame
        """
        dataset = self.load_generic(fpath, provided_df)

        if dataset_name in self.cfg.col_names:
            dataset = self.standardize_col_names(dataset, self.cfg.col_names[dataset_name])

        self.check_cols(dataset, dataset_name)

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
        cdr = self.load_dataset(
            dataset_name='cdr',
            fpath=fpath,
            provided_df=df
        )

        # Check txn_type column
        error_msg = 'CDR format incorrect. Column txn_type can only include call and text.'
        self.check_colvalues(cdr, 'txn_type', ['call', 'text'], error_msg)

        # Clean international column
        error_msg = 'CDR format incorrect. Column international can only include domestic, international, and other.'
        self.check_colvalues(cdr, 'international', ['domestic', 'international', 'other'], error_msg)

        # if no recipient antennas are present, add a null column to enable the featurizer to work
        if 'recipient_antenna' not in cdr.columns:
            cdr['recipient_antenna'] = None
            cdr['recipient_antenna'] = cdr['recipient_antenna'].astype('object')

        # Clean timestamp column
        cdr = self.clean_timestamp_and_add_day_column(cdr, 'timestamp')

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
        recharges = self.load_dataset('recharges', fpath=fpath, provided_df=df)

        # Clean timestamp column
        recharges = self.clean_timestamp_and_add_day_column(recharges, 'timestamp')

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
        mobiledata = self.load_dataset('mobiledata', fpath=fpath, provided_df=df)

        # Clean timestamp column
        mobiledata = self.clean_timestamp_and_add_day_column(mobiledata, 'timestamp')

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
        # load data as generic df and standardize column_names
        mobilemoney = self.load_dataset('mobilemoney', fpath=fpath, provided_df=df)

        # Check txn_type column
        txn_types = ['cashin', 'cashout', 'p2p', 'billpay', 'other']
        error_msg = 'Mobile money format incorrect. Column txn_type can only include ' + ', '.join(txn_types)
        self.check_colvalues(mobilemoney, 'txn_type', txn_types, error_msg)

        # Clean timestamp column
        mobilemoney = self.clean_timestamp_and_add_day_column(mobilemoney, 'timestamp')

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
        # Check the first row for time info, and assume the format is consistent
        existing_timestamp_sample = df[existing_timestamp_column_name].head(1).iloc[0]
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
        
        distinct_count = phone_numbers_to_featurize['phone_number'].nunique().compute()
        length = len(phone_numbers_to_featurize)

        if distinct_count != length:
            raise ValueError(
                f'Duplicates found in list of phone numbers to featurize: there are {distinct_count} distinct values '
                f'in a list of length {length}.'
            )

        return phone_numbers_to_featurize[['phone_number']]
