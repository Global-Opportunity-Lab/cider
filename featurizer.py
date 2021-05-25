from utils import *
from plot_utils import *
from cbio import *

class Featurizer:

    def __init__(self, wd, cdr_fname=None, topups_fname=None, mobiledata_fname=None, mobilemoney_fname=None):

        # Prepare working directory
        self.wd = wd
        make_dir(wd)
        make_dir(wd + '/tables')
        make_dir(wd + '/datasets')
        make_dir(wd + '/plots')

        # Load CDR data 
        self.cdr_fname = cdr_fname
        if cdr_fname is not None:
            print('Loading CDR...')
            self.cdr = load_cdr(cdr_fname)
        else:
            self.cdr = None

        # Load topups data
        self.topups_fname=topups_fname
        if topups_fname is not None:
            print('Loading topups...')
            self.topups = load_topups(topups_fname)
        else:
            self.topups=None

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
    
    def get_attr(self, attr):

        if attr == 'cdr':
            return self.cdr

        elif attr == 'topups':
            return self.topups

        elif attr == 'mobiledata':
            return self.mobiledata

        elif attr == 'mobilemoney':
            return self.mobilemoney
        
        else:
            raise ValueError(attr + ' is not a valid attribute.')
    
    def set_attr(self, attr, df):
        
        if attr == 'cdr':
            self.cdr = df

        elif attr == 'topups':
            self.topups = df

        elif attr == 'mobiledata':
            self.mobiledata = df

        elif attr == 'mobilemoney':
            self.mobilemoney = df
        
        else:
            raise ValueError(attr + ' is not a valid attribute.')


    def diagnostic_statistics(self, write=True):

        statistics = {}

        for name, df in [('CDR', self.cdr), 
                        ('Topups', self.topups), 
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
                        ('Topups', self.topups), 
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

        for df_name in ['cdr', 'topups', 'mobiledata', 'mobilemoney']:
            if self.get_attr(df_name) is not None:
                self.set_attr(df_name, filter_dates_dataframe(self.get_attr(df_name), start_date, end_date))
    

    def deduplicate(self):

        for df_name in ['cdr', 'topups', 'mobiledata', 'mobilemoney']:
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
        if self.topups is not None:
            self.topups = self.topups.where(~col('caller_id').isin(self.spammers))
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
        for df_name in ['cdr', 'topups', 'mobiledata', 'mobilemoney']:
            for outlier in outliers:
                outlier = pd.to_datetime(outlier)
                if self.get_attr(df_name) is not None:
                    self.set_attr(df_name, self.get_attr(df_name)\
                        .where((col('timestamp') < outlier) | (col('timestamp') >= outlier + pd.Timedelta(days=1))))
        
        return outliers

    def cdr_features(self):

        # Write parquet file in bandicoot format
        cols = ['txn_type', 'caller_id', 'recipient_id', 'timestamp', 'duration', 'caller_antenna', 'recipient_antenna']

        outgoing = self.cdr.select(cols)\
            .withColumnRenamed('txn_type', 'interaction')\
            .withColumnRenamed('caller_id', 'name')\
            .withColumnRenamed('recipient_id', 'correspondent_id')\
            .withColumnRenamed('timestamp', 'datetime')\
            .withColumnRenamed('duration', 'call_duration')\
            .withColumnRenamed('caller_antenna', 'antenna_id')\
            .withColumn('direction', lit('out'))\
            .drop('recipient_antenna')

        incoming = df.select(cols)\
            .withColumnRenamed('txn_type', 'interaction')\
            .withColumnRenamed('recipient_id', 'name')\
            .withColumnRenamed('caller_id', 'correspondent_id')\
            .withColumnRenamed('timestamp', 'datetime')\
            .withColumnRenamed('duration', 'call_duration')\
            .withColumnRenamed('recipient_antenna', 'antenna_id')\
            .withColumn('direction', lit('in'))\
            .drop('caller_antenna')

        records = outgoing.select(incoming.columns).union(incoming)\
            .withColumn('call_duration', col('call_duration').cast(IntegerType()))\
            .na.fill({'call_duration':'', 'correspondent_id':'', 'antenna_id':''} )
    
        save_parquet(records, wd + '/datasets/bandicoot_parquet')


        


        





        


        

        


        
        

        

        

        