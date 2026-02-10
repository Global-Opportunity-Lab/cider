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
import fiona
import datetime
import geopandas as gpd  # type: ignore[import]
from box import Box
from geopandas import GeoDataFrame
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import col, date_trunc, lit, to_timestamp, from_unixtime
from pyspark.sql.types import (StructType, StructField, StringType, 
                              TimestampType, FloatType, IntegerType, 
                              LongType, BooleanType, DoubleType)


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
        if 'read_options' not in self.cfg:
            self.cfg.read_options = {
                "timestampFormat": "yyyy-MM-dd HH:mm:ss"
            }

        
    def load_generic(
        self,
        fpath: Optional[Path] = None,
        df: Optional[Union[SparkDataFrame, PandasDataFrame]] = None,
        dataset_name: Optional[str] = None
    ) -> SparkDataFrame:

        schema = None
        if "data_schema" in self.cfg and dataset_name in ['cdr', 'mobiledata', 'mobilemoney', 'recharges']:
            if self.cfg.verbose >= 2:
                print(f"loading schema for {dataset_name}")
            schema_dict = self.cfg["data_schema"][dataset_name]
            fields = []
            
            type_mapping = {
                'string': StringType(),
                'timestamp': TimestampType(),
                'float': FloatType(),
                'integer': IntegerType(),
                'long': LongType(),
                'boolean': BooleanType(),
                'double': DoubleType(),
            }
            
            for field_name, field_type in schema_dict.items():
                spark_type = type_mapping.get(field_type.lower())
                if spark_type:
                    fields.append(StructField(field_name, spark_type, True))
                else:
                    raise ValueError(f"Unsupported type {field_type} for field {field_name}")
            
            schema = StructType(fields)

        # Load from file
        if fpath is not None:
            # The implicit assumption here is that the path falls under one of the following categories:
            # 1. A single parquet file
            # 2. A single csv file
            # 3. A directory containing only parquet files
            # 4. A directory containing only csv files
            parquet_files = list(Path(fpath).rglob('*.parquet'))
            csv_files = list(Path(fpath).rglob('*.csv'))
            if fpath.suffix == '.csv':
                df = self.spark.read.csv(str(fpath), header=True)
            elif fpath.suffix == '.parquet':
                df = self.spark.read.options(**self.cfg.read_options).schema(schema).parquet(str(fpath))
            elif Path(fpath).is_dir():
                if parquet_files:
                    parquet_paths = [str(path.resolve()) for path in parquet_files]
                    df = (self.spark.read
                        .options(**self.cfg.read_options)
                        .schema(schema)
                        .parquet(*parquet_paths))
                elif csv_files:
                    csv_path = str(Path(csv_files[0]).parent.resolve()) + "/*.csv"
                    df = (self.spark.read
                        .options(**self.cfg.read_options)
                        .schema(schema)
                        .csv(csv_path))
                else:
                    raise ValueError(f"No parquet or csv files found under directory: {fpath}")
        elif df is not None:
            if self.cfg.verbose >= 2:
                print(f"loading from provided df...")
            if not isinstance(df, SparkDataFrame):
                df = self.spark.createDataFrame(df)
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
                f"instead found {', '.join(columns_present)},"
                f"missing columns: {', '.join(required_cols - columns_present)}"
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
        col_mapping = {k: v for k, v in col_names.items()}
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
        dataset = self.load_generic(fpath, provided_df, dataset_name)
        if self.cfg.verbose >= 1:
            print(f"Dataset name: {dataset_name}")
        if dataset_name in self.cfg.col_names:
            if self.cfg.verbose >= 2:
                print(f"Standardizing column names for {dataset_name}")
            dataset = self.standardize_col_names(dataset, self.cfg.col_names[dataset_name])
        self.check_cols(dataset, dataset_name)
        return dataset    


    def load_cdr(
        self,
        fpath: Optional[Path] = None,
        df: Optional[Union[SparkDataFrame, PandasDataFrame]] = None,
        cfg: Optional[Box] = None
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
        # if no recipient antennas are present, add a null column to enable the featurizer to work
        # TODO(leo): Consider cleaning up featurizer logic so this isn't needed.
        if 'recipient_antenna' not in cdr.columns:
            cdr = cdr.withColumn('recipient_antenna', lit(None).cast(StringType()))

        cdr = self.clean_timestamp_and_add_day_column(cdr, 'timestamp')
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
        Load antennas' dataset, and print % of antennas that are missing coordinates.
        If fpath is a directory, will load and concatenate all valid files within it.

        Args:
            fpath: Path to antenna file or directory containing antenna files
            df: Optional dataframe to use instead of loading from file

        Returns: spark df with antenna data
        """
        parquet_files = list(Path(fpath).rglob('*.parquet'))
        csv_files = list(Path(fpath).rglob('*.csv'))
        fpath = Path(fpath)
        if fpath.suffix == '.csv':
            antennas = self.spark.read.csv(str(fpath), header=True)
        elif fpath.suffix == '.parquet':
            antennas = self.load_dataset('antennas', fpath=fpath)
        elif Path(fpath).is_dir():
            all_antennas = []
            if parquet_files:
                parquet_paths = [path.resolve() for path in parquet_files]
                for parquet_path in parquet_paths:
                    antenna_df = self.load_dataset('antennas', fpath=parquet_path, provided_df=None)
                    all_antennas.append(antenna_df)
            elif csv_files:
                csv_paths = [path.resolve() for path in csv_files]
                for csv_path in csv_paths:
                    antenna_df = self.load_dataset('antennas', fpath=csv_path, provided_df=None)
                    all_antennas.append(antenna_df)
            else:
                raise ValueError(f"No parquet or csv files found under directory: {fpath}")
            antennas = all_antennas[0]
            for df in all_antennas[1:]:
                antennas = antennas.unionAll(df)
        else:
            raise ValueError(f"No parquet or csv files found under directory: {fpath}")
        # Standardize column names and check required columns
        if 'antennas' in self.cfg.col_names:
            antennas = self.standardize_col_names(antennas, self.cfg.col_names['antennas'])
        self.check_cols(antennas, 'antennas')
        # Cast coordinates to float type
        antennas = antennas.withColumn('latitude', col('latitude').cast('float')) \
                          .withColumn('longitude', col('longitude').cast('float'))        
        number_missing_location = antennas.count() - antennas.select(['latitude', 'longitude']).na.drop().count()        
        if number_missing_location > 0:
            if self.cfg.verbose >= 1:
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
        recharges = self.clean_timestamp_and_add_day_column(recharges, 'timestamp')
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
        mobiledata = self.clean_timestamp_and_add_day_column(mobiledata, 'timestamp')
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
        """
        mobilemoney = self.load_dataset('mobilemoney', fpath=fpath, provided_df=df)
        mobilemoney = self.clean_timestamp_and_add_day_column(mobilemoney, 'timestamp')
        mobilemoney = mobilemoney.withColumn('amount', col('amount').cast('float'))
        for c in mobilemoney.columns:
            if 'balance' in c:
                mobilemoney = mobilemoney.withColumn(c, col(c).cast('float'))
        return mobilemoney


    def load_shapefile(self, fpath: Path) -> GeoDataFrame:
        """
        Load shapefile and make sure it has the right columns

        Args:
            fpath: path to file, which can be .shp, .zip, or .geojson

        Returns: GeoDataFrame
        """
        fpath_str = str(fpath)
        try:
            # For zip files, we need to specify the layer
            if fpath_str.endswith('.zip'):
                # List all layers in the zip file
                available_layers = fiona.listlayers(fpath_str)
                if not available_layers:
                    raise ValueError(f"No layers found in zip file: {fpath_str}")
                # Use the first layer by default
                shapefile = gpd.read_file(fpath_str, layer=available_layers[0])
            else:
                # For .shp and .geojson files, read directly
                shapefile = gpd.read_file(fpath_str)
            # Verify that columns are correct
            self.check_cols(shapefile, 'shapefile')
            # Verify that the geometry column has been loaded correctly
            assert shapefile.dtypes['geometry'] == 'geometry'
            shapefile['region'] = shapefile['region'].astype(str)
            return shapefile
        except Exception as e:
            print(f"Error loading shapefile {fpath_str}: {str(e)}")
            raise

    def clean_timestamp_and_add_day_column(
        self,
        df: SparkDataFrame,
        existing_timestamp_column_name: str,
    ):
        """
        Clean timestamp column and add a day column.
        
        Args:
            df: Spark DataFrame
            existing_timestamp_column_name: Name of the timestamp column
            
        Returns:
            DataFrame with cleaned timestamp and day column
        """
        # Get the data type of the timestamp column
        timestamp_type = df.schema[existing_timestamp_column_name].dataType
        
        # Check if it's already a timestamp type
        if isinstance(timestamp_type, TimestampType):
            # Already a timestamp, just add the day column
            return df.withColumn('timestamp', col(existing_timestamp_column_name)) \
                    .withColumn('day', date_trunc('day', col('timestamp')))
        
        # For numeric types (BIGINT, INT, etc.), convert from unix time
        elif isinstance(timestamp_type, (LongType, IntegerType)):
            divisor = getattr(self.cfg, 'unixtime_divisor', 1)
            timestamp_col = to_timestamp(from_unixtime(col(existing_timestamp_column_name) / divisor))
            return df.withColumn('timestamp', timestamp_col) \
                    .withColumn('day', date_trunc('day', col('timestamp')))
        
        # For string types, use to_timestamp directly
        elif isinstance(timestamp_type, StringType):
            return df.withColumn('timestamp', to_timestamp(col(existing_timestamp_column_name))) \
                    .withColumn('day', date_trunc('day', col('timestamp')))
        
        # For other types, try a sample-based approach as fallback
        else:
            sample_row = df.select(existing_timestamp_column_name).first()
            if sample_row is None:
                # Empty DataFrame
                return df.withColumn('timestamp', lit(None).cast(TimestampType())) \
                        .withColumn('day', lit(None).cast(TimestampType()))
            
            sample_timestamp = sample_row[0]
            
            if isinstance(sample_timestamp, (int, float)):
                # Numeric timestamp (unix timestamp)
                divisor = getattr(self.cfg, 'unixtime_divisor', 1)
                timestamp_col = to_timestamp(from_unixtime(col(existing_timestamp_column_name) / divisor))
                return df.withColumn('timestamp', timestamp_col) \
                        .withColumn('day', date_trunc('day', col('timestamp')))
            elif isinstance(sample_timestamp, str):
                # String timestamp
                return df.withColumn('timestamp', to_timestamp(col(existing_timestamp_column_name))) \
                        .withColumn('day', date_trunc('day', col('timestamp')))
            else:
                # Unknown type, try to cast to timestamp
                return df.withColumn('timestamp', col(existing_timestamp_column_name).cast(TimestampType())) \
                        .withColumn('day', date_trunc('day', col('timestamp')))


    def load_phone_numbers_to_featurize(
        self,
        fpath: Optional[Path] = None,
        df: Optional[Union[SparkDataFrame, PandasDataFrame]] = None,
    ) -> SparkDataFrame:
        phone_numbers_to_featurize = self.load_dataset(
            'phone_numbers_to_featurize', fpath=fpath, provided_df=df
        )
        return phone_numbers_to_featurize.select('phone_number')
