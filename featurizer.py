import bandicoot as bc
from box import Box
import yaml
import sys

from helpers.utils import *
from helpers.io_utils import *
from helpers.plot_utils import *



class Featurizer:

    def __init__(self, cfg_dir, clean_folders=False):
        # Read config file
        with open(cfg_dir, "r") as ymlfile:
            cfg = Box(yaml.safe_load(ymlfile))
        self.cfg = cfg
        data = cfg.path.featurizer.data
        outputs = cfg.path.featurizer.outputs
        self.outputs = outputs
        filenames = cfg.path.featurizer.filenames

        # Prepare working directory
        self.features = {'cdr': None, 'international': None, 'recharges': None,
                         'location': None, 'mobiledata': None, 'mobilemoney': None}
        make_dir(outputs, clean_folders)
        make_dir(outputs + '/tables')
        make_dir(outputs + '/datasets')
        make_dir(outputs + '/plots')

        # Spark setup
        spark = get_spark_session(self.cfg)
        self.spark = spark

        # Load CDR data 
        if filenames.cdr is not None:
            print('Loading CDR...')
            # Get fpath and load data
            fpath = data + filenames.cdr
            self.cdr = load_cdr(self.cfg, fpath)
            self.cdr_bandicoot = None
        else:
            self.cdr = None

        # Load antennas data 
        if filenames.antennas is not None:
            # Get fpath and load data
            fpath = data + filenames.antennas
            print('Loading antennas...')
            self.antennas = load_antennas(self.cfg, fpath)
        else:
            self.antennas = None

        # Load recharges data
        if filenames.recharges is not None:
            print('Loading recharges...')
            # Get fpath and load data
            fpath = data + filenames.recharges
            self.recharges = load_recharges(self.cfg, fpath)
        else:
            self.recharges=None

        # Load mobile internet data
        if filenames.mobiledata is not None:
            print('Loading mobile data...')
            # Get fpath and load data
            fpath = data + filenames.mobiledata
            self.mobiledata = load_mobiledata(self.cfg, fpath)
        else:
            self.mobiledata = None

        # Load mobile money data 
        if filenames.mobilemoney is not None:
            print('Loading mobile money...')
            # Get fpath and load data
            fpath = data + filenames.mobilemoney
            self.mobilemoney = load_mobilemoney(self.cfg, fpath)
        else:
            self.mobilemoney = None

        # Load shapefiles
        self.shapefiles = {}
        shapefiles = filenames.shapefiles
        for shapefile_fname in shapefiles.keys():
            self.shapefiles[shapefile_fname] = load_shapefile(data + shapefiles[shapefile_fname])

    def get_attr(self, attr):

        if attr == 'cdr':
            return self.cdr
        
        elif attr == 'antennas':
            return self.antennas

        elif attr == 'recharges':
            return self.recharges

        elif attr == 'mobiledata':
            return self.mobiledata

        elif attr == 'mobilemoney':
            return self.mobilemoney
        
        else:
            raise ValueError(attr + ' is not a valid attribute.')
    
    def set_attr(self, attr, df):
        
        if attr == 'cdr':
            self.cdr = df
        
        elif attr == 'antennas':
            self.antennas = df

        elif attr == 'recharges':
            self.recharges = df

        elif attr == 'mobiledata':
            self.mobiledata = df

        elif attr == 'mobilemoney':
            self.mobilemoney = df
        
        else:
            raise ValueError(attr + ' is not a valid attribute.')

    def diagnostic_statistics(self, write=True):

        statistics = {}

        for name, df in [('CDR', self.cdr), 
                        ('Recharges', self.recharges), 
                        ('Mobile Data', self.mobiledata), 
                        ('Mobile Money', self.mobilemoney)]:
            if df is not None:

                statistics[name] = {}

                # Number of days
                lastday = pd.to_datetime(df.agg({'timestamp':'max'}).collect()[0][0])
                firstday = pd.to_datetime(df.agg({'timestamp':'min'}).collect()[0][0])
                statistics[name]['Days'] = (lastday - firstday).days + 1

                # Number of transactions
                statistics[name]['Transactions'] = df.count()

                # Number of subscribers
                statistics[name]['Subscribers'] = df.select('caller_id').distinct().count()

                # Number of recipients
                if 'recipient_id' in df.columns:
                    statistics[name]['Recipients'] = df.select('recipient_id').distinct().count()
        
        if write:
            with open(self.outputs + '/tables/statistics.json', 'w') as f:
                json.dump(statistics, f)

        return statistics


    def diagnostic_plots(self, plot=True):

        for name, df in [('CDR', self.cdr), 
                        ('Recharges', self.recharges), 
                        ('Mobile Data', self.mobiledata), 
                        ('Mobile Money', self.mobilemoney)]:
            if df is not None:

                if 'txn_type' not in df.columns:
                    df = df.withColumn('txn_type', lit('txn'))

                # Save timeseries of transactions by day
                save_df(df.groupby(['txn_type', 'day']).count(), self.outputs + '/datasets/' + name.replace(' ', '') + '_transactionsbyday.csv')

                # Save timeseries of subscribers by day
                save_df(df.groupby(['txn_type', 'day']).agg(countDistinct('caller_id')).withColumnRenamed('count(caller_id)', 'count'), \
                    self.outputs + '/datasets/' + name.replace(' ', '') + '_subscribersbyday.csv')

                if plot:

                    # Plot timeseries of transactions by day
                    timeseries = pd.read_csv(self.outputs + '/datasets/' + name.replace(' ', '') + '_transactionsbyday.csv')
                    timeseries['day'] = pd.to_datetime(timeseries['day'])
                    timeseries = timeseries.sort_values('day', ascending=True)
                    fig, ax = plt.subplots(1, figsize=(20, 6))
                    for txn_type in timeseries['txn_type'].unique():
                        subset = timeseries[timeseries['txn_type'] == txn_type]
                        ax.plot(subset['day'], subset['count'], label=txn_type)
                        ax.scatter(subset['day'], subset['count'], label='')
                    if len(timeseries['txn_type'].unique()) > 1:
                        ax.legend(loc='best')
                    ax.set_title(name + ' Transactions by Day', fontsize='large')
                    dates_xaxis(ax, frequency='week')
                    clean_plot(ax)
                    plt.savefig(self.outputs + '/plots/' + name.replace(' ', '') + '_transactionsbyday.png', dpi=300)

                    # Plot timeseries of subscribers by day
                    timeseries = pd.read_csv(self.outputs + '/datasets/' + name.replace(' ', '') + '_subscribersbyday.csv')
                    timeseries['day'] = pd.to_datetime(timeseries['day'])
                    timeseries = timeseries.sort_values('day', ascending=True)
                    fig, ax = plt.subplots(1, figsize=(20, 6))
                    for txn_type in timeseries['txn_type'].unique():
                        subset = timeseries[timeseries['txn_type'] == txn_type]
                        ax.plot(subset['day'], subset['count'], label=txn_type)
                        ax.scatter(subset['day'], subset['count'], label='')
                    if len(timeseries['txn_type'].unique()) > 1:
                        ax.legend(loc='best')
                    ax.set_title(name + ' Subscribers by Day', fontsize='large')
                    dates_xaxis(ax, frequency='week')
                    clean_plot(ax)
                    plt.savefig(self.outputs + '/plots/' + name.replace(' ', '') + '_subscribersbyday.png', dpi=300)


    def filter_dates(self, start_date, end_date):

        for df_name in ['cdr', 'recharges', 'mobiledata', 'mobilemoney']:
            if self.get_attr(df_name) is not None:
                self.set_attr(df_name, filter_dates_dataframe(self.get_attr(df_name), start_date, end_date))


    def deduplicate(self):

        for df_name in ['cdr', 'recharges', 'mobiledata', 'mobilemoney']:
            if self.get_attr(df_name) is not None:
                self.set_attr(df_name, self.get_attr(df_name).distinct())


    def remove_spammers(self, spammer_threshold=100):

        # Raise exception if no CDR, since spammers are calculated only on the basis of call and text
        if self.cdr is None:
            raise ValueError('CDR must be loaded to identify and remove spammers.')

        # Get average number of calls and SMS per day
        grouped = (self.cdr
                   .groupby('caller_id', 'txn_type')
                   .agg(count(lit(0)).alias('n_transactions'),
                        countDistinct(col('day')).alias('active_days'))
                   .withColumn('count', col('n_transactions')/col('active_days')))

        # Get list of spammers
        self.spammers = grouped.where(col('count') > spammer_threshold).select('caller_id').distinct().rdd.map(lambda r: r[0]).collect()
        pd.DataFrame(self.spammers).to_csv(self.outputs + '/datasets/spammers.csv', index=False)
        print('Number of spammers identified: %i' % len(self.spammers))

        # Remove transactions (incoming or outgoing) associated with spammers from all dataframes
        self.cdr = self.cdr.where(~col('caller_id').isin(self.spammers))
        self.cdr = self.cdr.where(~col('recipient_id').isin(self.spammers))
        if self.recharges is not None:
            self.recharges = self.recharges.where(~col('caller_id').isin(self.spammers))
        if self.mobiledata is not None:
            self.mobiledata = self.mobiledata.where(~col('caller_id').isin(self.spammers))
        if self.mobilemoney is not None:
            self.mobilemoney = self.mobilemoney.where(~col('caller_id').isin(self.spammers))
            self.mobilemoney = self.mobilemoney.where(~col('recipient_id').isin(self.spammers))
        
        return self.spammers
        
    def filter_outlier_days(self, num_sds=2):

        # Raise exception if no CDR, since spammers are calculated only on the basis of call and text
        if self.cdr is None:
            raise ValueError('CDR must be loaded to identify and remove outlier days.')

        # If haven't already obtained timeseries of subscribers by day (e.g. in diagnostic plots), calculate it
        if not os.path.isfile(self.outputs + '/datasets/CDR_transactionsbyday.csv'):
            save_df(self.cdr.groupby(['txn_type', 'day']).count(), self.outputs + '/datasets/CDR_transactionsbyday.csv')

        # Read in timeseries of subscribers by day
        timeseries = pd.read_csv(self.outputs + '/datasets/CDR_transactionsbyday.csv')

        # Calculate timeseries of all transaction (voice + SMS together)
        timeseries = timeseries.groupby('day', as_index=False).agg('sum')

        # Calculate top and bottom acceptable values
        bottomrange = timeseries['count'].mean() - num_sds*timeseries['count'].std()
        toprange = timeseries['count'].mean() + num_sds*timeseries['count'].std()

        # Obtain list of outlier days
        outliers = timeseries[(timeseries['count'] < bottomrange) | (timeseries['count'] > toprange)]
        outliers.to_csv(self.outputs + '/datasets/outlier_days.csv', index=False)
        outliers = list(outliers['day'])
        print('Outliers removed: ' + ', '.join([outlier.split('T')[0] for outlier in outliers]))

        # Remove outlier days from all datasets 
        for df_name in ['cdr', 'recharges', 'mobiledata', 'mobilemoney']:
            for outlier in outliers:
                outlier = pd.to_datetime(outlier)
                if self.get_attr(df_name) is not None:
                    self.set_attr(df_name, self.get_attr(df_name)\
                        .where((col('timestamp') < outlier) | (col('timestamp') >= outlier + pd.Timedelta(days=1))))
        
        return outliers

    def cdr_features(self, bc_chunksize=500000, bc_processes=55):

        # Check that CDR is present to calculate international features
        if self.cdr is None:
            raise ValueError('CDR file must be loaded to calculate CDR features.')
        print('Calculating CDR features...')

        # Convert CDR into bandicoot format
        self.cdr_bandicoot = cdr_bandicoot_format(self.cdr, self.antennas, self.cfg.cdr.col_names)

        # Get list of unique subscribers, write to file
        save_df(self.cdr_bandicoot.select('name').distinct(), self.outputs + '/datasets/subscribers.csv')
        subscribers = self.cdr_bandicoot.select('name').distinct().rdd.map(lambda r: r[0]).collect()

        # Make adjustments to chunk size and parallelization if necessary
        if bc_chunksize > len(subscribers):
            bc_chunksize = len(subscribers)
        if bc_processes > int(len(subscribers)/bc_chunksize):
            bc_processes = int(len(subscribers)/bc_chunksize)

        # Make output folders
        make_dir(self.outputs + '/datasets/bandicoot_records')
        make_dir(self.outputs + '/datasets/bandicoot_features')

        # Get bandicoot features in chunks
        start = 0
        end = 0
        while end < len(subscribers):
            
            # Get start and end point of chunk
            end = start + bc_chunksize
            chunk = subscribers[start:end]

            # Name outfolders
            recs_folder = self.outputs + '/datasets/bandicoot_records/' + str(start) + 'to' + str(end)
            bc_folder = self.outputs + '/datasets/bandicoot_features/' + str(start) + 'to' + str(end)
            make_dir(bc_folder)

            # Get records for this chunk and write out to csv files per person
            nums_spark = self.spark.createDataFrame(chunk, StringType()).withColumnRenamed('value', 'name')
            matched_chunk = self.cdr_bandicoot.join(nums_spark, on='name', how='inner')
            matched_chunk.repartition('name').write.partitionBy('name').mode('append').format('csv').save(recs_folder, header=True)

            # Move csv files around on disk to get into position for bandicoot
            n = int(len(chunk)/bc_processes)
            subchunks = [chunk[i:i+n] for i in range(0, len(chunk), n)]
            pool = Pool(bc_processes)
            unmatched = pool.map(flatten_folder, [(subchunk, recs_folder) for subchunk in subchunks])
            unmatched = flatten_lst(unmatched)
            pool.close()
            if len(unmatched) > 0:
                print('Warning: lost %i subscribers in file shuffling' % len(unmatched))

            # Calculate bandicoot features
            def get_bc(sub):
                return bc.utils.all(bc.read_csv(str(sub), recs_folder, describe=True), summary='extended', split_week=True, \
                    split_day=True, groupby=None)

            # Write out bandicoot feature files
            def write_bc(index, iterator):
                bc.to_csv(list(iterator), bc_folder +  '/' + str(index) + '.csv')
                return ['index: ' + str(index)]

            # Run calculations and writing of bandicoot features in parallel
            feature_df = self.spark.sparkContext.emptyRDD()
            subscriber_rdd = self.spark.sparkContext.parallelize(chunk)
            features = subscriber_rdd.mapPartitions(lambda s: [get_bc(sub) for sub in s if os.path.isfile(recs_folder + '/' + sub + '.csv')])
            feature_df = feature_df.union(features)
            out = feature_df.coalesce(bc_processes).mapPartitionsWithIndex(write_bc)
            out.count()
            start = start + bc_chunksize
        
        # Combine all bandicoot features into a single file, fix column names, and write to disk
        cdr_features = self.spark.read.csv(self.outputs + '/datasets/bandicoot_features/*/*', header=True)
        cdr_features = cdr_features.select([col for col in cdr_features.columns if ('reporting' not in col) or (col == 'reporting__number_of_records')])
        cdr_features = cdr_features.toDF(*[c if c == 'name' else 'cdr_' + c for c in cdr_features.columns])
        save_df(cdr_features, self.outputs + '/datasets/bandicoot_features/all.csv')
        self.features['cdr'] = self.spark.read.csv(self.outputs + '/datasets/bandicoot_features/all.csv', header=True)

    def international_features(self):

        # Check that CDR is present to calculate international features
        if self.cdr is None:
            raise ValueError('CDR file must be loaded to calculate international features.')
        print('Calculating international features...')

        # Write international transactions to file
        international_trans = self.cdr.filter(col('international') == 'international')
        save_df(international_trans, self.outputs + '/datasets/internatonal_transactions.csv')
    
        # Read international calls
        inter = pd.read_csv(self.outputs + '/datasets/internatonal_transactions.csv')

        # Calculate list of aggregations by subscriber
        inter_voice = inter[inter['txn_type'] == 'call']
        inter_sms = inter[inter['txn_type'] == 'text']
        lst = [
            ('recipient_id', ['count', 'nunique']),
            ('day', ['nunique']),
            ('duration', ['sum'])
        ]
        feats = []
        for c, agg in lst:
            for subset, name in [(inter, 'all'), (inter_voice, 'call'), (inter_sms, 'text')]:
                grouped = subset[['caller_id', c]].groupby('caller_id', as_index=False).agg(agg)
                grouped.columns = [name + '__' + c + '__' + ag for ag in agg]
                feats.append(grouped)
                
        # Combine all aggregations together, write to file
        feats = long_join_pandas(feats, on='caller_id', how='outer').rename({'caller_id':'name'}, axis=1)
        feats['name'] = feats.index
        feats.columns = [c if c == 'name' else 'international_' + c for c in feats.columns]
        feats.to_csv(self.outputs + '/datasets/international_feats.csv', index=False)
        self.features['international'] = self.spark.read.csv(self.outputs + '/datasets/international_feats.csv', header=True)

    
    def location_features(self):

        # Check that antennas and CDR are present to calculate spatial features
        if self.cdr is None:
            raise ValueError('CDR file must be loaded to calculate spatial features.')
        if self.antennas is None:
            raise ValueError('Antenna file must be loaded to calculate spatial features.')
        print('Calculating spatial features...')

        # If CDR is not available in bandicoot format, calculate it
        if self.cdr_bandicoot is None:
            self.cdr_bandicoot = cdr_bandicoot_format(self.cdr, self.antennas)

        # Get dataframe of antennas located within regions
        antennas = pd.read_csv(self.antennas_fname)
        antennas = gpd.GeoDataFrame(antennas, geometry=gpd.points_from_xy(antennas['longitude'], antennas['latitude']))
        antennas.crs = {"init":"epsg:4326"}
        for shapefile_name in self.shapefiles.keys():
            shapefile = self.shapefiles[shapefile_name].rename({'region':shapefile_name}, axis=1)
            antennas = gpd.sjoin(antennas, shapefile, op='within', how='left').drop('index_right', axis=1)
            antennas[shapefile_name] = antennas[shapefile_name].fillna('Unknown')
        antennas = self.spark.createDataFrame(antennas.drop(['geometry', 'latitude', 'longitude'], axis=1).fillna(''))
        
        # Merge CDR to antennas
        cdr = self.cdr_bandicoot.join(antennas, on='antenna_id', how='left')\
            .na.fill({shapefile_name:'Unknown' for shapefile_name in self.shapefiles.keys()})

        # Get counts by region and unique regions
        for shapefile_name in self.shapefiles.keys():
            countbyregion = cdr.groupby(['name', shapefile_name]).count()
            save_df(countbyregion, self.outputs + '/datasets/countby' + shapefile_name + '.csv')

        # Get unique regions (and unique towers)
        unique_regions = cdr.select('name').distinct()
        for shapefile_name in self.shapefiles.keys():
            unique_regions = unique_regions.join(cdr.groupby('name').agg(countDistinct(shapefile_name)), on='name', how='left')
        if 'tower_id' in cdr.columns:
            unique_regions = unique_regions.join(cdr.groupby('name').agg(countDistinct('tower_id')), on='name', how='left')
        save_df(unique_regions, self.outputs + '/datasets/uniqueregions.csv')

        # Pivot counts by region
        count_by_region_compiled = []
        for shapefile_name in self.shapefiles.keys():
            count_by_region = pd.read_csv(self.outputs + '/datasets/countby' + shapefile_name + '.csv')\
                .pivot(index='name', columns=shapefile_name, values='count').fillna(0)
            count_by_region['total'] = count_by_region.sum(axis=1)
            for c in set(count_by_region.columns) - set(['total', 'name']):
                count_by_region[c + '_percent'] = count_by_region[c]/count_by_region['total']
            count_by_region = count_by_region.rename({region:shapefile_name + '_' + region for region in count_by_region.columns}, axis=1)
            count_by_region_compiled.append(count_by_region)
        
        count_by_region = long_join_pandas(count_by_region_compiled, on='name', how='outer')
        count_by_region = count_by_region.drop([c for c in count_by_region.columns if 'total' in c], axis=1)
        
        # Read in the unique regions
        unique_regions = pd.read_csv(self.outputs + '/datasets/uniqueregions.csv')

        # Merge counts and unique counts together, write to file
        feats = count_by_region.merge(unique_regions, on='name', how='outer')
        feats.columns = [c if c == 'name' else 'location_' + c for c in feats.columns]
        feats.to_csv(self.outputs + '/datasets/location_features.csv', index=False)
        self.features['location'] = self.spark.read.csv(self.outputs + '/datasets/location_features.csv', header=True)
        

    def mobiledata_features(self):

        # Check that mobile internet data is loaded
        if self.mobiledata is None:
            raise ValueError('Mobile data file must be loaded to calculate mobile data features.')
        print('Calculating mobile data features...')

        # Perform set of aggregations on mobile data 
        feats = self.mobiledata.groupby('caller_id').agg(sum('volume').alias('total_volume'), 
                                                        mean('volume').alias('mean_volume'),
                                                        min('volume').alias('min_volume'),
                                                        max('volume').alias('max_volume'),
                                                        stddev('volume').alias('std_volume'),
                                                        countDistinct('day').alias('num_days'),
                                                        count('volume').alias('num_transactions'))

        # Save to file
        feats = feats.withColumnRenamed('caller_id', 'name')
        feats = feats.toDF(*[c if c == 'name' else 'mobiledata_' + c for c in feats.columns])
        self.features['mobiledata'] = feats
        save_df(feats, self.outputs + '/datasets/mobiledata_features.csv')

    def mobilemoney_features(self):

        # Check that mobile money is loaded
        if self.mobilemoney is None:
            raise ValueError('Mobile money file must be loaded to calculate mobile money features.')
        print('Calculating mobile money features...')

        # Get outgoing transactions
        sender_cols = ['txn_type', 'caller_id', 'recipient_id', 'day', 'amount', 'sender_balance_before',
                       'sender_balance_after']
        outgoing = (self.mobilemoney
                    .select(sender_cols)
                    .withColumnRenamed('caller_id', 'name')
                    .withColumnRenamed('recipient_id', 'correspondent_id')
                    .withColumnRenamed('sender_balance_before', 'balance_before')
                    .withColumnRenamed('sender_balance_after', 'balance_after')
                    .withColumn('direction', lit('out')))

        # Get incoming transactions
        recipient_cols = ['txn_type', 'caller_id', 'recipient_id', 'day', 'amount', 'recipient_balance_before',
                          'recipient_balance_after']
        incoming = (self.mobilemoney.select(recipient_cols)
                    .withColumnRenamed('recipient_id', 'name')
                    .withColumnRenamed('caller_id', 'correspondent_id')
                    .withColumnRenamed('recipient_balance_before', 'balance_before')
                    .withColumnRenamed('recipient_balance_after', 'balance_after')
                    .withColumn('direction', lit('in')))

        # Combine incoming and outgoing with unified schema
        mm = outgoing.select(incoming.columns).union(incoming)
        save_parquet(mm, self.outputs + '/datasets/mobilemoney')
        mm = self.spark.read.parquet(self.outputs + '/datasets/mobilemoney')
        outgoing = mm.where(col('direction') == 'out')
        incoming = mm.where(col('direction') == 'in')

        # Get mobile money features
        features = []
        txn_types = mm.select('txn_type').distinct().rdd.map(lambda r: r[0]).collect()
        for dfname, df in [('all', mm), ('incoming', incoming), ('outgoing', outgoing)]:
            # add 'all' txn type
            df = (df
                  .withColumn('txn_types', array(lit('all'), col('txn_type')))
                  .withColumn('txn_type', explode('txn_types')))

            aggs = (df
                    .groupby('name', 'txn_type')
                    .agg(mean('amount').alias('amount_mean'),
                         min('amount').alias('amount_min'),
                         max('amount').alias('amount_max'),
                         mean('balance_before').alias('balance_before_mean'),
                         min('balance_before').alias('balance_before_min'),
                         max('balance_before').alias('balance_before_max'),
                         mean('balance_after').alias('balance_after_mean'),
                         min('balance_after').alias('balance_after_min'),
                         max('balance_after').alias('balance_after_max'),
                         count('correspondent_id').alias('txns'),
                         countDistinct('correspondent_id').alias('contacts'))
                    .groupby('name')
                    .pivot('txn_type')
                    .agg(first('amount_mean').alias('amount_mean'),
                         first('amount_min').alias('amount_min'),
                         first('amount_max').alias('amount_max'),
                         first('balance_before_mean').alias('balance_before_mean'),
                         first('balance_before_min').alias('balance_before_min'),
                         first('balance_before_max').alias('balance_before_max'),
                         first('balance_after_mean').alias('balance_after_mean'),
                         first('balance_after_min').alias('balance_after_min'),
                         first('balance_after_max').alias('balance_after_max'),
                         first('txns').alias('txns'),
                         first('contacts').alias('contacts')))
            # add df name to columns
            for col_name in aggs.columns[1:]: # exclude 'name'
                aggs = aggs.withColumnRenamed(col_name, dfname + '_' + col_name)

            features.append(aggs)

        # Combine all mobile money features together and save them
        feats = long_join_pyspark(features, on='name', how='outer')
        feats = feats.toDF(*[c if c == 'name' else 'mobilemoney_' + c for c in feats.columns])
        save_df(feats, self.outputs '/datasets/mobilemoney_feats.csv')
        self.features['mobilemoney'] = self.spark.read.csv(self.outputs + '/datasets/mobilemoney_feats.csv', header=True)

    
    def recharges_features(self):

        if self.recharges is None:
            raise ValueError('Recharges file must be loaded to calculate recharges features.')
        print('Calculating recharges features...')

        feats = self.recharges.groupby('caller_id').agg(sum('amount').alias('sum'),
                                                        mean('amount').alias('mean'),
                                                        min('amount').alias('min'),
                                                        max('amount').alias('max'),
                                                        count('amount').alias('count'),
                                                        countDistinct('day').alias('days'))

        feats = feats.withColumnRenamed('caller_id', 'name')
        feats = feats.toDF(*[c if c == 'name' else 'recharges_' + c for c in feats.columns])
        save_df(feats, self.outputs + '/datasets/recharges_feats.csv')
        self.features['recharges'] = self.spark.read.csv(self.outputs '/datasets/recharges_feats.csv', header=True)


    def all_features(self):

        all_features = [self.features[key] for key in self.features.keys() if self.features[key] is not None]
        all_features = long_join_pyspark(all_features, how='left', on='name')
        save_df(all_features, self.outputs + '/datasets/features.csv')
        self.features['all'] = self.spark.read.csv(self.outputs + '/datasets/features.csv', header=True)


    def feature_plots(self, try_disk=False, data_path=None):
        if not data_path:
            data_path = self.outputs + '/datasets/'

        features = ['cdr', 'international', 'location', 'mobiledata', 'mobilemoney', 'recharges']
        datasets = ['all', 'international_feats', 'location_features', 'mobiledata_features', 'mobilemoney_feats', 'recharges_feats']
        # Read data from disk if requested
        if try_disk:
            for feature, dataset in zip(features, datasets):
                if not self.features[feature]:
                    try:
                        self.features[feature] = self.spark.read.csv(data_path + dataset + '.csv', header=True)
                    except:
                        print(f"Could not locate or read data for '{dataset}'")

        # Plot of distributions of CDR features
        if self.features['cdr'] is not None:
            features = ['cdr_active_days__allweek__day__callandtext',  'cdr_call_duration__allweek__allday__call__mean', 'cdr_number_of_antennas__allweek__allday']
            names = ['Active Days', 'Mean Call Duration', 'Number of Antennas']
            distributions_plot(self.features['cdr'], features, names, color='indianred')
            plt.savefig(self.outputs + '/plots/cdr.png', dpi=300)
            plt.show()

        # Plot of distributions of international features
        if self.features['international'] is not None:
            features = ['international_all__recipient_id__count', 'international_all__recipient_id__nunique', 'international_call__duration__sum']
            names = ['International Transactions', 'International Contaacts', 'Total International Call Time']
            distributions_plot(self.features['international'], features, names, color='darkorange')
            plt.savefig(self.outputs + '/plots/international.png', dpi=300)
            plt.show()

        # Plot of distributions of recharges features
        if self.features['recharges'] is not None:
            features = ['recharges_mean', 'recharges_count', 'recharges_days']
            names = ['Mean Recharge Amount', 'Number of Recharges', 'Number of Days with Recharges']
            distributions_plot(self.features['recharges'], features, names, color='mediumseagreen')
            plt.savefig(self.outputs + '/plots/recharges.png', dpi=300)
            plt.show()
        
        # Plot of distributions of mobile data features
        if self.features['mobiledata'] is not None:
            features = ['mobiledata_total_volume', 'mobiledata_mean_volume', 'mobiledata_num_days']
            names = ['Total Volume (MB)', 'Mean Transaction Volume (MB)', 'Number of Days with Data Usage']
            distributions_plot(self.features['mobiledata'], features, names, color='dodgerblue')
            plt.savefig(self.outputs + '/plots/mobiledata.png', dpi=300)
            plt.show()

        # Plot of distributions of mobile money features
        if self.features['mobilemoney'] is not None:
            features = ['mobilemoney_all_all_amount_mean', 'mobilemoney_all_all_balance_before_mean', 'mobilemoney_all_all_txns', 'mobilemoney_all_cashout_txns']
            names = ['Mean Amount', 'Mean Balance', 'Transactions', 'Cashout Transactions']
            distributions_plot(self.features['mobilemoney'], features, names, color='orchid')
            plt.savefig(self.outputs + '/plots/mobilemoney.png', dpi=300)
            plt.show()

        # Spatial plots
        if self.features['location'] is not None:
            for shapefile_name in self.shapefiles.keys():
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                columns = [c for c in self.features['location'].columns if shapefile_name in c and 'percent' not in c and 'Unknown' not in c]
                counts = self.features['location'].select([sum(c) for c in columns]).toPandas()
                counts.columns = ['_'.join(c.split('_')[2:])[:-1] for c in counts.columns]
                counts = counts.T
                counts.columns = ['txn_count']
                counts['region'] = counts.index
                counts = self.shapefiles[shapefile_name].merge(counts, on='region', how='left')
                counts['txn_count'] = counts['txn_count'].fillna(0)/counts['txn_count'].sum()
                counts.plot(ax=ax, column='txn_count', cmap='magma', legend=True, legend_kwds={'shrink':0.5})
                ax.axis('off')
                ax.set_title('Proportion of Transactions by ' + shapefile_name, fontsize='large')
                plt.tight_layout()
                plt.savefig(self.outputs + '/plots/spatial_' + shapefile_name + '.png')
                plt.show()

        # Cuts by feature usage (mobile money, mobile data, international calls)
        if self.features['cdr'] is not None:

            all_subscribers  = self.features['cdr'].select('name')

            if self.features['international'] is not None:
                international_subscribers = self.features['international'].where(col('international_all__recipient_id__count') > 0).select('name')
            else:
                international_subscribers = None
            
            if self.features['mobiledata'] is not None:
                mobiledata_subscribers = self.features['mobiledata'].where(col('mobiledata_num_transactions') > 0).select('name')
            else:
                mobiledata_subscribers = None

            if self.features['mobilemoney'] is not None:
                mobilemoney_subscribers = self.features['mobilemoney'].where(col('mobilemoney_all_all_txns') > 0).select('name')
            else:
                mobilemoney_subscribers = None

            features = ['cdr_active_days__allweek__day__callandtext',  'cdr_call_duration__allweek__allday__call__mean', 'cdr_number_of_antennas__allweek__allday']
            names = ['Active Days', 'Mean Call Duration', 'Number of Antennas']

            fig, ax = plt.subplots(1, len(features), figsize=(20, 5))
            for a in range(len(features)):
                boxplot = []
                for subscribers, slice_name in [(all_subscribers, 'All'),
                                                (international_subscribers, 'I Callers'), 
                                                (mobiledata_subscribers, 'MD Users'), 
                                                (mobilemoney_subscribers, 'MM Users')]:
                    if subscribers is not None:
                        users = self.features['cdr'].join(subscribers, how='inner', on='name')
                        slice = users.select(['name', features[a]]).toPandas()
                        slice['slice_name'] = slice_name
                        boxplot.append(slice)
                boxplot = pd.concat(boxplot)
                boxplot[features[a]] = boxplot[features[a]].astype('float')
                sns.boxplot(data=boxplot, x=features[a], y='slice_name', ax=ax[a], palette="Set2", orient='h')
                ax[a].set_xlabel('Feature')
                ax[a].set_ylabel(names[a])
                ax[a].set_title(names[a], fontsize='large')
                clean_plot(ax[a])
            plt.savefig(self.outputs + '/plots/boxplots.png', dpi=300)
            plt.show()


























































































