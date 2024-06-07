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

import geopandas as gpd  # type: ignore[import]
from box import Box
from geopandas import GeoDataFrame
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import col, date_trunc, lit, to_timestamp
from pyspark.sql.types import StringType


from helpers.utils import get_spark_session

class IOUtils:
    
    def __init__(
        self,
        cfg: Box, 
        data_format: Box
    ):
        self.cfg = cfg
        self.data_format = data_format
        self.spark = get_spark_session(cfg)

    
    def load_generic(
        self,
        fpath: Optional[Path] = None,
        df: Optional[Union[SparkDataFrame, PandasDataFrame]] = None
    ) -> SparkDataFrame:

        # Load from file
        if fpath is not None:
            # Load data if in a single file
            if fpath.is_file():

                if fpath.suffix == '.csv':
                    df = self.spark.read.csv(str(fpath), header=True)

                elif fpath.suffix == '.parquet':
                    df = self.spark.read.parquet(str(fpath))

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
                    df = self.spark.read.csv(str(fpath / '*.csv'), header=True)

                elif example_file.suffix == '.parquet':
                    df = self.spark.read.parquet(str(fpath / '*.parquet'), header=True)

                else:
                    raise ValueError(f'Found file with unknown extension {example_file}.')

        # Load from pandas dataframe
        elif df is not None:
            if not isinstance(df, SparkDataFrame):
                df = spark.createDataFrame(df)

        # Issue with filename/dataframe provided
        else:
            raise ValueError('No filename or pandas/spark dataframe provided.')

        return df


    def check_cols(
        self,
        df: Union[GeoDataFrame, PandasDataFrame, SparkDataFrame],
        dataset_name: str,
    ) -> None:
        """
        Check that the df has all required columns

        Args:
            df: spark df
            dataset_name: name of dataset, to be used in error messages.
            dataset_data_format: box containing data format information.
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
        self, df: SparkDataFrame, colname: str, colvalues: list, error_msg: str
    ) -> None:
        """
        Check that a column has all required values

        Args:
            df: spark df
            colname: column to check
            colvalues: requires values
            error_msg: error message to print if values don't match
        """
        if set(df.select(colname).distinct().rdd.map(lambda r: r[0]).collect()).union(set(colvalues)) != set(colvalues):
            raise ValueError(error_msg)


    def standardize_col_names(
        self, df: SparkDataFrame, col_names: Dict[str, str]
    ) -> SparkDataFrame:
        """
        Rename columns, as specified in config file, to standard format

        Args:
            df: spark df
            col_names: mapping between standard column names and existing ones

        Returns: spark df with standardized column names

        """
        col_mapping = {v: k for k, v in col_names.items()}

        for col in df.columns:
            if col in col_mapping:
                df = df.withColumnRenamed(col, col_mapping[col])

        return df

    # TODO: Rename to "load_generic", rename load_generic to "load_from_disk" or something
    def load_dataset(
        self,
        dataset_name: str,
        fpath: Optional[Path] = None,
        provided_df: Optional[Union[SparkDataFrame, PandasDataFrame]] = None
    ) -> SparkDataFrame:

        dataset = self.load_generic(fpath, provided_df)

        if dataset_name in self.cfg.col_names:
            dataset = self.standardize_col_names(dataset, self.cfg.col_names[dataset_name])

        self.check_cols(dataset, dataset_name)

        return dataset    


    def load_cdr(
        self,
        fpath: Optional[Path] = None,
        df: Optional[Union[SparkDataFrame, PandasDataFrame]] = None
    ) -> SparkDataFrame:
        """
        Load CDR data into spark df

        Returns: spark df
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
        # TODO(leo): Consider cleaning up featurizer logic so this isn't needed.
        if 'recipient_antenna' not in cdr.columns:
            cdr = cdr.withColumn('recipient_antenna', lit(None).cast(StringType()))

        # Clean timestamp column
        cdr = self.clean_timestamp_and_add_day_column(cdr, 'timestamp')

        # Clean duration column
        cdr = cdr.withColumn('duration', col('duration').cast('float'))

        return cdr


    def load_labels(
        self,
        fpath: Path = None
    ) -> SparkDataFrame:

        """
        Load labels on which to train ML model.
        """

        labels = self.load_dataset('labels', fpath=fpath)

        if 'weight' not in labels.columns:
            labels = labels.withColumn('weight', lit(1))

        return labels.select(['name', 'label', 'weight'])


    def load_antennas(
        self,
        fpath: Optional[Path] = None,
        df: Optional[Union[SparkDataFrame, PandasDataFrame]] = None
    ) -> SparkDataFrame:
        """
        Load antennas' dataset, and print % of antennas that are missing coordinates

        Returns: spark df
        """

        antennas = self.load_dataset('antennas', fpath=fpath, provided_df = df)

        antennas = antennas.withColumn('latitude', col('latitude').cast('float')).withColumn('longitude',
                                                                                             col('longitude').cast(
                                                                                                 'float'))
        
        number_missing_location = antennas.count() - antennas.select(['latitude', 'longitude']).na.drop().count()
        
        if number_missing_location > 0:
            print(f'Warning: {number_missing_location} antennas missing location')

        return antennas


    def load_recharges(
        self,
        fpath: Optional[Path] = None,
        df: Optional[Union[SparkDataFrame, PandasDataFrame]] = None
    ) -> SparkDataFrame:
        """
        Load recharges dataset

        Returns: spark df
        """

        recharges = self.load_dataset('recharges', fpath=fpath, provided_df=df)

        # Clean timestamp column
        recharges = self.clean_timestamp_and_add_day_column(recharges, 'timestamp')

        # Clean amount column
        recharges = recharges.withColumn('amount', col('amount').cast('float'))

        return recharges


    def load_mobiledata(
        self,
        fpath: Optional[Path] = None,
        df: Optional[Union[SparkDataFrame, PandasDataFrame]] = None
    ) -> SparkDataFrame:
        """
        Load mobile data dataset

        """

        mobiledata = self.load_dataset('mobiledata', fpath=fpath, provided_df=df)

        # Clean timestamp column
        mobiledata = self.clean_timestamp_and_add_day_column(mobiledata, 'timestamp')

        # Clean duration column
        mobiledata = mobiledata.withColumn('volume', col('volume').cast('float'))

        return mobiledata


    def load_mobilemoney(
        self,
        fpath: Optional[Path] = None,
        df: Optional[Union[SparkDataFrame, PandasDataFrame]] = None,
        verify: bool = True
    ) -> SparkDataFrame:
        """
        Load mobile money dataset

        Returns: spark df
        """

        # load data as generic df and standardize column_names
        mobilemoney = self.load_dataset('mobilemoney', fpath=fpath, provided_df=df)

         # Check txn_type column
        txn_types = ['cashin', 'cashout', 'p2p', 'billpay', 'other']
        error_msg = 'Mobile money format incorrect. Column txn_type can only include ' + ', '.join(txn_types)
        self.check_colvalues(mobilemoney, 'txn_type', txn_types, error_msg)

        # Clean timestamp column
        mobilemoney = self.clean_timestamp_and_add_day_column(mobilemoney, 'timestamp')

        # Clean duration column
        mobilemoney = mobilemoney.withColumn('amount', col('amount').cast('float'))

        # Clean balance columns
        for c in mobilemoney.columns:
            if 'balance' in c:
                mobilemoney = mobilemoney.withColumn(c, col(c).cast('float'))

        return mobilemoney


    def load_shapefile(self, fpath: Path) -> GeoDataFrame:
        """
        Load shapefile and make sure it has the right columns

        Args:
            fpath: path to file, which can be .shp or .geojson

        Returns: GeoDataFrame

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
        df: SparkDataFrame,
        existing_timestamp_column_name: str
    ):

        # Check the first row for time info, and assume the format is consistent
        existing_timestamp_sample = df.take(1)[0][existing_timestamp_column_name]
        timestamp_with_time_regex = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"

        has_time_info = bool(re.match(timestamp_with_time_regex, existing_timestamp_sample))

        timestamp_format = (
            'yyyy-MM-dd HH:mm:ss' if has_time_info else 'yyyy-MM-dd'
        )

        return (
            df
            .withColumn(
                'timestamp',
                to_timestamp(existing_timestamp_column_name, timestamp_format)
            )
            .withColumn('day', date_trunc('day', col('timestamp')))
        )


    def load_phone_numbers_to_featurize(
        self,
        fpath: Optional[Path] = None,
        df: Optional[Union[SparkDataFrame, PandasDataFrame]] = None,
    ) -> SparkDataFrame:

        phone_numbers_to_featurize = self.load_dataset(
            'phone_numbers_to_featurize', fpath=fpath, provided_df=df
        )
        
        distinct_count = phone_numbers_to_featurize.select(col('phone_number')).distinct().count()
        length = phone_numbers_to_featurize.count()

        if distinct_count != length:
            raise ValueError(
                f'Duplicates found in list of phone numbers to featurize: there are {distinct_count} distinct values '
                f'in a list of length {length}.'
            )

        return phone_numbers_to_featurize.select('phone_number')
