# TODO: 
# Improve efficiency of mobile money aggregations
# Plots
# Tests
# Combine features together, fix up column naming

from typing import cast
from utils import *
from plot_utils import *
from cbio import *
import bandicoot as bc

class Featurizer:

    def __init__(self, wd, cdr_fname=None, antennas_fname=None, recharges_fname=None, mobiledata_fname=None, mobilemoney_fname=None, 
    shapefiles=[], interim_write=True):

        # Prepare working directory
        self.wd = wd
        self.interim_write = interim_write
        self.features = {}
        make_dir(wd)
        make_dir(wd + '/tables')
        make_dir(wd + '/datasets')
        make_dir(wd + '/plots')

        # Load CDR data 
        self.cdr_fname = cdr_fname
        if cdr_fname is not None:
            print('Loading CDR...')
            self.cdr = load_cdr(cdr_fname)
            self.cdr_bandicoot = None

        else:
            self.cdr = None

        # Load antennas data 
        self.antennas_fname = antennas_fname
        if antennas_fname is not None:
            print('Loading antennas...')
            self.antennas = load_antennas(antennas_fname)
        else:
            self.cdr = None

        # Load recharges data
        self.recharges_fname=recharges_fname
        if recharges_fname is not None:
            print('Loading recharges...')
            self.recharges = load_recharges(recharges_fname)
        else:
            self.recharges=None

        # Load mobile internet data
        self.mobiledata_fname=mobiledata_fname
        if mobiledata_fname is not None:
            print('Loading mobile data...')
            self.mobiledata = load_mobiledata(mobiledata_fname)
        else:
            self.mobiledata = None

        # Load mobile money data 
        self.mobilemoney_fname=mobilemoney_fname
        if mobilemoney_fname is not None:
            print('Loading mobile money...')
            self.mobilemoney = load_mobilemoney(mobilemoney_fname)
        else:
            self.mobilemoney = None

        # Load shapefiles
        self.shapefiles = {}
        for shapefile_fname in shapefiles.keys():
            self.shapefiles[shapefile_fname] = load_shapefile(shapefiles[shapefile_fname])
    
    
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
            with open(self.wd + '/tables/statistics.json', 'w') as f:
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
                save_df(df.groupby(['txn_type', 'day']).count(), self.wd + '/datasets/' + name.replace(' ', '') + '_transactionsbyday.csv')

                # Save timeseries of subscribers by day
                save_df(df.groupby(['txn_type', 'day']).agg(countDistinct('caller_id')).withColumnRenamed('count(caller_id)', 'count'), \
                    self.wd + '/datasets/' + name.replace(' ', '') + '_subscribersbyday.csv')

                if plot:

                    # Plot timeseries of transactions by day
                    timeseries = pd.read_csv(self.wd + '/datasets/' + name.replace(' ', '') + '_transactionsbyday.csv')
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
                    plt.savefig(self.wd + '/plots/' + name.replace(' ', '') + '_transactionsbyday.png', dpi=300)

                    # Plot timeseries of subscribers by day
                    timeseries = pd.read_csv(self.wd + '/datasets/' + name.replace(' ', '') + '_subscribersbyday.csv')
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
                    plt.savefig(self.wd + '/plots/' + name.replace(' ', '') + '_subscribersbyday.png', dpi=300)


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

        # Get the number of days in the dataset
        lastday = pd.to_datetime(self.cdr.agg({'timestamp':'max'}).collect()[0][0])
        firstday = pd.to_datetime(self.cdr.agg({'timestamp':'min'}).collect()[0][0])
        ndays = (lastday - firstday).days + 1
        
        # Get count of calls and SMS per subscriber
        grouped = self.cdr.groupby(['caller_id', 'txn_type']).count()
        grouped = grouped.withColumn('count', col('count')/(ndays))

        # Get list of spammers
        self.spammers = grouped.where(col('count') > spammer_threshold).select('caller_id').distinct().rdd.map(lambda r: r[0]).collect()
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
        if not os.path.isfile(self.wd + '/datasets/CDR_transactionsbyday.csv'):
            save_df(self.cdr.groupby(['txn_type', 'day']).count(), self.wd + '/datasets/CDR_transactionsbyday.csv')

        # Read in timeseries of subscribers by day
        timeseries = pd.read_csv(self.wd + '/datasets/CDR_transactionsbyday.csv')

        # Calculate timeseries of all transaction (voice + SMS together)
        timeseries = timeseries.groupby('day', as_index=False).agg('sum')

        # Calculate top and bottom acceptable values
        bottomrange = timeseries['count'].mean() - num_sds*timeseries['count'].std()
        toprange = timeseries['count'].mean() + num_sds*timeseries['count'].std()

        # Obtain list of outlier days
        outliers = timeseries[(timeseries['count'] < bottomrange) | (timeseries['count'] > toprange)]
        outliers.to_csv(self.wd + '/datasets/outlier_days.csv', index=False)
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

        spark = get_spark_session()

        # Check that CDR is present to calculate international features
        if self.cdr is None:
            raise ValueError('CDR file must be loaded to calculate CDR features.')
        print('Calculating CDR features...')

        # Convert CDR into bandicoot format
        self.cdr_bandicoot = cdr_bandicoot_format(self.cdr, self.antennas)

        # Get list of unique subscribers, write to file
        save_df(self.cdr_bandicoot.select('name').distinct(), self.wd + '/datasets/subscribers.csv')
        subscribers = self.cdr_bandicoot.select('name').distinct().rdd.map(lambda r: r[0]).collect()

        # Make adjustments to chunk size and parallelization if necessary
        if bc_chunksize > len(subscribers):
            bc_chunksize = len(subscribers)
        if bc_processes > int(len(subscribers)/bc_chunksize):
            bc_processes = int(len(subscribers)/bc_chunksize)

        # Make output folders
        make_dir(self.wd + '/datasets/bandicoot_records')
        make_dir(self.wd + '/datasets/bandicoot_features')

        # Get bandicoot features in chunks
        start = 0
        end = 0
        while end < len(subscribers):
            
            # Get start and end point of chunk
            end = start + bc_chunksize
            chunk = subscribers[start:end]

            # Name outfolders
            recs_folder = self.wd + '/datasets/bandicoot_records/' + str(start) + 'to' + str(end)
            bc_folder = self.wd + '/datasets/bandicoot_features/' + str(start) + 'to' + str(end)
            make_dir(bc_folder)

            # Get records for this chunk and write out to csv files per person
            nums_spark = spark.createDataFrame(chunk, StringType()).withColumnRenamed('value', 'name')
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
            feature_df = spark.sparkContext.emptyRDD()
            subscriber_rdd = spark.sparkContext.parallelize(chunk)
            features = subscriber_rdd.mapPartitions(lambda s: [get_bc(sub) for sub in s if os.path.isfile(recs_folder + '/' + sub + '.csv')])
            feature_df = feature_df.union(features)
            out = feature_df.coalesce(bc_processes).mapPartitionsWithIndex(write_bc)
            out.count()
            start = start + bc_chunksize
        
        # Combine all bandicoot features into a single file, fix column names, and write to disk
        cdr_features = spark.read.csv(self.wd + '/datasets/bandicoot_features/*/*', header=True)
        cdr_features = cdr_features.select([col for col in cdr_features.columns if ('reporting' not in col) or (col == 'reporting__number_of_records')])
        cdr_features = cdr_features.toDF(*[c if c == 'name' else 'cdr_' + c for c in cdr_features.columns])
        save_df(cdr_features, self.wd + '/datasets/bandicoot_features/all.csv')
        self.features['cdr'] = cdr_features
        


    def international_features(self):

        spark = get_spark_session()

        # Check that CDR is present to calculate international features
        if self.cdr is None:
            raise ValueError('CDR file must be loaded to calculate international features.')
        print('Calculating international features...')

        # Write international transactions to file
        international_trans = self.cdr.filter(col('international') == 'international')
        save_df(international_trans, self.wd + '/datasets/internatonal_transactions.csv')
    
        # Read international calls
        inter = pd.read_csv(self.wd + '/datasets/internatonal_transactions.csv')

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
        feats.to_csv(self.wd + '/datasets/international_feats.csv', index=False)
        feats = spark.createDataFrame(feats)
        self.features['international'] = feats

    
    def location_features(self):

        spark = get_spark_session()

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
        for shapefile_name in self.shapefiles.keys():
            shapefile = self.shapefiles[shapefile_name].rename({'region':shapefile_name}, axis=1)
            antennas = gpd.sjoin(antennas, shapefile, op='within', how='left').drop('index_right', axis=1)
            antennas[shapefile_name] = antennas[shapefile_name].fillna('Unknown')
        antennas = spark.createDataFrame(antennas.drop(['geometry', 'latitude', 'longitude'], axis=1).fillna(''))
        
        # Merge CDR to antennas
        cdr = self.cdr_bandicoot.join(antennas, on='antenna_id', how='left')\
            .na.fill({shapefile_name:'Unknown' for shapefile_name in self.shapefiles.keys()})

        # Get counts by region and unique regions
        for shapefile_name in self.shapefiles.keys():
            countbyregion = cdr.groupby(['name', shapefile_name]).count()
            save_df(countbyregion, self.wd + '/datasets/countby' + shapefile_name + '.csv')

        # Get unique regions (and unique towers)
        unique_regions = cdr.select('name').distinct()
        for shapefile_name in self.shapefiles.keys():
            unique_regions = unique_regions.join(cdr.groupby('name').agg(countDistinct(shapefile_name)), on='name', how='left')
        if 'tower_id' in cdr.columns:
            unique_regions = unique_regions.join(cdr.groupby('name').agg(countDistinct('tower_id')), on='name', how='left')
        save_df(unique_regions, self.wd + '/datasets/uniqueregions.csv')

        # Pivot counts by region
        count_by_region_compiled = []
        for shapefile_name in self.shapefiles.keys():
            count_by_region = pd.read_csv(self.wd + '/datasets/countby' + shapefile_name + '.csv')\
                .pivot(index='name', columns=shapefile_name, values='count').fillna(0)
            count_by_region['total'] = count_by_region.sum(axis=1)
            for c in set(count_by_region.columns) - set(['total', 'name']):
                count_by_region[c + '_percent'] = count_by_region[c]/count_by_region['total']
            count_by_region = count_by_region.rename({region:shapefile_name + '_' + region for region in count_by_region.columns}, axis=1)
            count_by_region_compiled.append(count_by_region)
        
        count_by_region = long_join_pandas(count_by_region_compiled, on='name', how='outer')
        count_by_region = count_by_region.drop([c for c in count_by_region.columns if 'total' in c], axis=1)
        
        # Read in the unique regions
        unique_regions = pd.read_csv(self.wd + '/datasets/uniqueregions.csv')

        # Merge counts and unique counts together, write to file
        feats = count_by_region.merge(unique_regions, on='name', how='outer')
        feats.columns = [c if c == 'name' else 'location_' + c for c in feats.columns]
        feats.to_csv(self.wd + '/datasets/location_features.csv', index=False)
        self.features['location'] = spark.createDataFrame(feats)
        

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
        feats = feats.withColumnRenamed('caller_id', 'name')\
            .toDF(*[c if c == 'name' else 'mobiledata_' + c for c in feats.columns])
        self.features['mobiledata'] = feats
        save_df(feats, self.wd + '/datasets/mobiledata_feats.csv')

    def mobilemoney_features(self):

        spark = get_spark_session()

        # Check that mobile money is loaded
        if self.mobilemoney is None:
            raise ValueError('Mobile money file must be loaded to calculate mobile money features.')
        print('Calculating mobile money features...')

        # Get outgoing transactions
        sender_cols = ['txn_type', 'caller_id', 'recipient_id', 'day', 'amount', 'sender_balance_before', 'sender_balance_after']
        outgoing = self.mobilemoney.select(sender_cols)\
                .withColumnRenamed('caller_id', 'name')\
                .withColumnRenamed('recipient_id', 'correspondent_id')\
                .withColumnRenamed('sender_balance_before', 'balance_before')\
                .withColumnRenamed('sender_balance_after', 'balance_after')\
                .withColumn('direction', lit('out'))

        # Get incoming transactions
        recipient_cols = ['txn_type', 'caller_id', 'recipient_id', 'day', 'amount', 'recipient_balance_before', 'recipient_balance_after']
        incoming = self.mobilemoney.select(recipient_cols)\
                .withColumnRenamed('recipient_id', 'name')\
                .withColumnRenamed('caller_id', 'correspondent_id')\
                .withColumnRenamed('recipient_balance_before', 'balance_before')\
                .withColumnRenamed('recipient_balance_after', 'balance_after')\
                .withColumn('direction', lit('in'))

        # Combine incoming and outgoing with unified schema
        cols = ['txn_type', 'name', 'correspondent_id', 'balance_before', 'balance_after', 'day', 'amount', 'direction']
        mm = outgoing.select(incoming.columns).union(incoming)
        save_parquet(mm, self.wd + '/datasets/mobilemoney')
        mm = spark.read.parquet(self.wd + '/datasets/mobilemoney')
        outgoing = mm.where(col('direction') == 'out')
        incoming = mm.where(col('direction') == 'in')

        # Get mobile money features
        features = []
        for dfname, df in [('all', mm), ('incoming', incoming), ('outgoing', outgoing)]:
            for txn_type in mm.select('txn_type').distinct().rdd.map(lambda r: r[0]).collect():

                # Filter if restricting to certain transaction type
                if txn_type != 'all':
                    df_filtered = df.where(col('txn_type') == txn_type)

                # Aggregate columns relating to amount (amount, balance before, balance after) with min, median, and max
                for c in ['amount', 'balance_before', 'balance_after']:
                    colname_tag = dfname + '_' + txn_type + '_' + c + '_'
                    amount_aggs = df_filtered.groupby('name').agg(mean(c).alias(colname_tag + 'mean'), 
                                                                min(c).alias(colname_tag + 'min'), 
                                                                max(c).alias(colname_tag + 'max'))
                    features.append(amount_aggs)

                # Aggregate columns relating to count (number of transactions and number of distinct contacts)
                colname_tag = dfname + '_' + txn_type + '_'
                count_aggs = df_filtered.groupby('name').agg(count('correspondent_id').alias(colname_tag + 'txns'), 
                                                    countDistinct('correspondent_id').alias(colname_tag + 'contacts'))
                features.append(count_aggs)

        # Combine all mobile money features together and save them
        feats = long_join_pyspark(features, on='name', how='outer')
        feats = feats.toDF(*[c if c == 'name' else 'mobilemoney_' + c for c in feats.columns])
        self.features['mobilemoney'] = feats
        save_df(feats, self.wd + '/datasets/mobilemoney_feats.csv')

    
    def recharges_features(self):

        if self.recharges is None:
            raise ValueError('Recharges file must be loaded to calculate recharges features.')
        print('Calculating recharges features...')

        feats = self.recharges.groupby('caller_id').agg(sum('amount').alias('recharges_sum'),
                                                        mean('amount').alias('recharges_mean'),
                                                        min('amount').alias('recharges_min'),
                                                        max('amount').alias('recharges_max'),
                                                        count('amount').alias('recharges_count'),
                                                        countDistinct('day').alias('recharges_days'))

        feats = feats.withColumnRenamed('caller_id', 'name')\
            .toDF(*[c if c == 'name' else 'recharges_' + c for c in feats.columns])
        self.features['recharges'] = feats
        save_df(feats, self.wd + '/datasets/recharges_feats.csv')


    def all_features(self):

        all_features = [self.features[key] for key in self.features.keys() if self.features[key] is not None]
        all_features = long_join_pyspark(all_features, how='left', on='name')
        save_df(all_features, self.wd + '/dataset/features.csv')


        



                

        

        
        






        















            








        


        


        





        


        

        


        
        

        

        

        