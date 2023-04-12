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

import warnings
from collections import defaultdict
from numbers import Number
from typing import Callable, Dict, Iterable, List, Optional, Union

from hashids import Hashids
from numpy import isnan
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

from helpers.utils import make_dir, save_df

from .datastore import DataStore, DataType


class Anonymizer:

    def __init__(
        self, 
        datastore: DataStore, 
        dataframes: Optional[Dict[str, Optional[Union[PandasDataFrame, SparkDataFrame]]]] = None,
        clean_folders = True, 
        format_checker: Optional[Callable] = None):
        """
        Args:
            clean_folders: Whether to clear the output directory if it already exists
            format_checker: A custom callable to check phone number length, format, prefix, etc. It's important to
              ensure consistency when anonymizing identifiers; for example, the addition/omission of a
              leading zero will result in a completely different obfuscated value.

        Returns: dict of dicts containing summary stats - {'CDR': {'Transactions': 2.3, ...}, ...}   
        """
        self.ds=datastore
        self.cfg = datastore.cfg
        self.outputs_path = self.cfg.path.working.directory_path / 'anonymized'

        make_dir(self.outputs_path, clean_folders)
        make_dir(self.outputs_path / 'outputs')

        with open(self.cfg.path.input_data.file_paths.anonymization_salt, 'r') as saltfile:
            salt = saltfile.read().strip() 

        self.encoder = Hashids(salt=salt, min_length=16)
        self.format_checker = format_checker

        dataframes = dataframes if dataframes else defaultdict(lambda: None)

        data_type_map = {
            DataType.LABELS: dataframes['labels'],
            DataType.CDR: dataframes['cdr'],
            DataType.RECHARGES: dataframes['recharges'],
            DataType.MOBILEDATA: dataframes['mobiledata'],
            DataType.MOBILEMONEY: dataframes['mobilemoney'],
            DataType.FEATURES: dataframes['features']
        }

        self.ds.load_data(data_type_map=data_type_map, all_required=False)


    def anonymize_cdr(self):
        self._anonymize_dataset('cdr', ['caller_id', 'recipient_id'])


    def anonymize_mobilemoney(self):
        self._anonymize_dataset('mobilemoney', ['caller_id', 'recipient_id'])


    def anonymize_mobiledata(self):
        self._anonymize_dataset('mobiledata', 'caller_id')


    def anonymize_recharges(self):
        self._anonymize_dataset('recharges', 'caller_id')

    
    def anonymize_features(self):
        self._anonymize_dataset('features', 'name')


    def anonymize_labels(self):
        self._anonymize_dataset('labels', 'name')


    def _anonymize_dataset(self, dataset_name: str, column_names: List) -> None:

        try:
            dataset = getattr(self.ds, dataset_name)
        except AttributeError:
            raise ValueError(f'Dataset {dataset_name}. Perhaps no path is specified in config.')
        
        if isinstance(column_names, str):
            column_names = [column_names]
        
        encoder = self.encoder
        format_checker = self.format_checker
        
        dataset_with_anonymized_columns = dataset
        
        # reverse iteration order to get the new columns in the specified order, at the beginning
        # of the dataframe.
        for column_name in reversed(column_names):
            
            if column_name in dataset.columns:
                
                new_column = udf(
                    lambda raw: Anonymizer._check_identifier_format_and_hash(raw, encoder, format_checker), StringType()
                )(dataset[column_name]).alias(f'{column_name}_anonymized')
                
                # using the select function (instead of withColumn) places the new column at the beginning of the df
                dataset_with_anonymized_columns = (
                    dataset_with_anonymized_columns.select(new_column, '*').drop(column_name)
                )

        save_df(dataset_with_anonymized_columns, self.outputs_path / 'outputs' / f'{dataset_name}.csv')

        
    # this function is static to allow passing a reference to spark workers
    @staticmethod
    def _check_identifier_format_and_hash(raw, encoder, format_checker):
        
        # missing values pass through
        if raw is None:
            return None
        
        # input can be an actual integer
        elif isinstance(raw, Number):
            
            # missing values pass through
            if isnan(raw):
                return None
            
            raw_int = int(raw)
            
            # this will evaluate true on integers or integral floats (e.g. int(5.0) == 5.0 is true)
            if raw_int == raw:
                raw_string = str(raw_int)
            else:
                raise ValueError(
                    f'Bad input to anonymization: {raw} is a non-integer number, which cannot represent a phone number.'
                )
            
        # input can be a string representing an integer
        else:
            raw_string = str(raw).strip('+')
            if raw_string.isdigit():
                raw_int = int(raw)

            else:
                print(raw)
                raise ValueError(f'Bad input to anonymization: {raw} (type {type(raw)}) is not an integer or a string of digits.')

        # evaluate custom format_checker if provided
        if format_checker and not format_checker(raw_string):
            raise ValueError(f'Bad input to anonymization: {raw_string} rejected by provided format format_checker.')

        return encoder.encode(raw_int)
