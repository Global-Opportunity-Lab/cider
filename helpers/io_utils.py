from helpers.utils import *
import geopandas as gpd


def load_generic(cfg, fname=None, df=None):
	spark = get_spark_session(cfg)

	# Load from file
	if fname is not None:
		# Load data if in a single file
		if '.csv' in fname:
			df = spark.read.csv(fname, header=True)

		# Load data if in chunks
		else:
			df = spark.read.csv(fname + '/*.csv', header=True)
	
	# Load from pandas dataframe
	elif df is not None:
		df = spark.createDataFrame(df)
	
	# Issue with filename/dataframe provided
	else:
		print('No filename or pandas dataframe provided.')
		return ''
	
	return df


def check_cols(df, required_cols, error_msg):
	if not set(required_cols).intersection(set(df.columns)) == set(required_cols):
		raise ValueError(error_msg)


def check_colvalues(df, colname, colvalues, error_msg):
	if set(df.select(colname).distinct().rdd.map(lambda r: r[0]).collect()).union(set(colvalues)) != set(colvalues):
		raise ValueError(error_msg)


def standardize_col_names(df, col_names):
	col_mapping = {v: k for k, v in col_names.items()}

	for col in df.columns:
		df = df.withColumnRenamed(col, col_mapping[col])

	return df
	

def load_cdr(cfg, fname=None, df=None, verify=True):
	# load data as generic df and standardize column_names
	cdr = load_generic(cfg, fname=fname, df=df)
	cdr = standardize_col_names(cdr, cfg.col_names.cdr)

	if verify:
	
		# Check that required columns are present
		required_cols = ['txn_type', 'caller_id', 'recipient_id', 'timestamp', 'duration']
		error_msg = 'CDR format incorrect. CDR must include the following columns: ' + ', '.join(required_cols)
		check_cols(cdr, required_cols, error_msg)

		# Check txn_type column
		error_msg = 'CDR format incorrect. Column txn_type can only include call and text.'
		check_colvalues(cdr, 'txn_type', ['call', 'text'], error_msg)
		
		# Clean international column
		error_msg = 'CDR format incorrect. Column international can only include domestic, international, and other.'
		check_colvalues(cdr, 'international', ['domestic', 'international', 'other'], error_msg)
	
	# Clean timestamp column
	cdr = cdr.withColumn('timestamp', to_timestamp(cdr['timestamp'], 'yyyy-MM-dd HH:mm:ss'))\
		.withColumn('day', date_trunc('day', col('timestamp')))
		
	# Clean duration column
	cdr = cdr.withColumn('duration', col('duration').cast('float'))
	
	return cdr


def load_antennas(cfg, fname=None, df=None, verify=True):
	# load data as generic df and standardize column_names
	antennas = load_generic(cfg, fname=fname, df=df)
	antennas = standardize_col_names(antennas, cfg.col_names.antennas)

	if verify:

		required_cols = ['antenna_id', 'latitude', 'longitude']
		error_msg = 'Antenna format incorrect. Antenna dataset must include the following columns: ' + ', '.join(required_cols)
		check_cols(antennas, required_cols, error_msg)

		antennas = antennas.withColumn('latitude', col('latitude').cast('float')).withColumn('longitude', col('longitude').cast('float'))
		print('Warning: %i antennas missing location' % (antennas.count() - antennas.select(['latitude', 'longitude']).na.drop().count()))
	
	return antennas


def load_recharges(cfg, fname=None, df=None, verify=True):
	# load data as generic df and standardize column_names
	recharges = load_generic(cfg, fname=fname, df=df)
	recharges = standardize_col_names(recharges, cfg.col_names.recharges)
		
	# Clean timestamp column
	recharges = recharges.withColumn('timestamp', to_timestamp(recharges['timestamp'], 'yyyy-MM-dd HH:mm:ss'))\
		.withColumn('day', date_trunc('day', col('timestamp')))
		
	# Clean duration column
	recharges = recharges.withColumn('amount', col('amount').cast('float'))
		
	return recharges


def load_mobiledata(cfg, fname=None, df=None, verify=True):
	# load data as generic df and standardize column_names
	mobiledata = load_generic(cfg, fname=fname, df=df)
	mobiledata = standardize_col_names(mobiledata, cfg.col_names.mobiledata)
		
	# Clean timestamp column
	mobiledata = mobiledata.withColumn('timestamp', to_timestamp(mobiledata['timestamp'], 'yyyy-MM-dd HH:mm:ss'))\
		.withColumn('day', date_trunc('day', col('timestamp')))
	
	# Clean duration column
	mobiledata = mobiledata.withColumn('volume', col('volume').cast('float'))
	
	return mobiledata


def load_mobilemoney(cfg, fname=None, df=None, verify=True):
	# load data as generic df and standardize column_names
	mobilemoney = load_generic(cfg, fname=fname, df=df)
	mobilemoney = standardize_col_names(mobilemoney, cfg.col_names.mobilemoney)

	if verify:
	
		# Check that required columns are present
		required_cols = ['txn_type', 'caller_id', 'recipient_id', 'timestamp', 'amount']
		error_msg = 'Mobile money format incorrect. Mobile money records must include the following columns: ' + \
			', '.join(required_cols)
		check_cols(mobilemoney, required_cols, error_msg)
		
		# Check txn_type column
		txn_types = ['cashin', 'cashout', 'p2p', 'billpay', 'other']
		error_msg = 'Mobile money format incorrect. Column txn_type can only include ' + ', '.join(txn_types)
		check_colvalues(mobilemoney, 'txn_type', txn_types, error_msg)

	# Clean timestamp column
	mobilemoney = mobilemoney.withColumn('timestamp', to_timestamp(mobilemoney['timestamp'], 'yyyy-MM-dd HH:mm:ss'))\
		.withColumn('day', date_trunc('day', col('timestamp')))
	
	# Clean duration column
	mobilemoney = mobilemoney.withColumn('amount', col('amount').cast('float'))

	# Clean balance columns
	for c in mobilemoney.columns:
		if 'balance' in c:
			mobilemoney = mobilemoney.withColumn(c, col(c).cast('float'))
	
	return mobilemoney


def load_shapefile(fname):
	shapefile = gpd.read_file(fname) 

	# Verify that columns are correct
	required_cols = ['region', 'geometry']
	error_msg = 'Shapefile format incorrect. Shapefile must include the following columns: ' +  ', '.join(required_cols)
	check_cols(shapefile, required_cols, error_msg)

	return shapefile