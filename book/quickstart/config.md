# Configuration file

The configuration file - example at `configs/config.yml` - is used to store all relevant configurations, like paths to \
the datasets and spark parameters. It should be appropriately edited before executing the code. <br>

The first parameters to set are those related to spark:

```
spark: 
  app_name: "mm" 
  files:
    max_partition_bytes: 67108864
  driver:
    memory: "8g" // driver memory
    max_result_size: "2g" // maximum result size when collecting to driver
  loglevel: "ERROR"
```

Then we set the paths to the folders that store our files, and also specify the file names:

```
path:
  data: "/Users/example/Documents/GD/cider/synthetic_data/"
  features: '/Users/example/Documents/GD/cider/outputs/featurizer/datasets/features.csv'
  file_names:
    antennas: 'antennas.csv'
    cdr: 'cdr.csv'
    home_ground_truth: 'home_locations.csv'
    labels: 'labels.csv'
    mobiledata: 'mobiledata.csv'
    mobilemoney: 'mobilemoney.csv'
    population: 'population_tgo_2019-07-01.tif'
    poverty_scores: null
    recharges: 'recharges.csv'
    rwi: '/TGO_relative_wealth_index.csv'
    shapefiles:
      regions: 'regions.geojson'
      cantons: 'cantons.geojson'
      prefectures: 'prefectures.geojson'
    user_consent: null
  outputs: "/Users/example/Documents/GD/cider/outputs/"  // output folder
  wd: "/Users/example/Documents/GD/cider/"  // working directory
```

The featurizer module expects certain column and column names, and we can define them in the following section of the 
config file:

```
col_names:
  cdr:
    txn_type: "txn_type"
    caller_id: "caller_id"
    recipient_id: "recipient_id"
    timestamp: "timestamp"
    duration: "duration"
    caller_antenna: "caller_antenna"
    recipient_antenna: "recipient_antenna"
    international: "international"
  antennas:
    antenna_id: "antenna_id"
    tower_id: "tower_id"
    latitude: "latitude"
    longitude: "longitude"
  recharges:
    caller_id: "caller_id"
    amount: "amount"
    timestamp: "timestamp"
  mobiledata:
    caller_id: "caller_id"
    volume: "volume"
    timestamp: "timestamp"
  mobilemoney:
    txn_type: "txn_type"
    caller_id: "caller_id"
    recipient_id: "recipient_id"
    timestamp: "timestamp"
    amount: "amount"
    sender_balance_before: "sender_balance_before"
    sender_balance_after: "sender_balance_after"
    recipient_balance_before: "recipient_balance_before"
    recipient_balance_after: "recipient_balance_after"
  geo: 'tower_id'
```

We also have to set a few parameters that will affect the behaviour of some modules:

```
params:
  cdr:
    weekend: [1, 7] // definition of weekend (Sun = 1 to Sat = 7)
    start_of_day: 7 // hour when day starts (used to define day/night)
    end_of_day: 19 // hour when night starts (used to define day/night)
  home_location:
    filter_hours: null // hours to filter out when inferring home locations
  automl: // params used by the autoML libraries
    autosklearn:
      time_left: 3600
      n_jobs: 1
      memory_limit: 3072
    autogluon:
      time_limit: 3600
      eval_metric: 'r2'
      label: 'label'
      sample_weight: 'weight'
  opt_in_default: false // if true opt-in is set as default, i.e. all users give their consent unless they opt-out
```

Finally, we can set the hyper-parameters that will be tested during a grid-search performed by the ML module"

```
hyperparams:
  'linear':
    'dropmissing__threshold': [0.9, 1]
    'droplowvariance__threshold': [ 0, 0.01 ]
    'winsorizer__limits': [!!python/tuple [0., 1.], !!python/tuple [0.005, .995]]
  'lasso':
    'dropmissing__threshold': [ 0.9, 1 ]
    'droplowvariance__threshold': [ 0, 0.01 ]
    'winsorizer__limits': [!!python/tuple [0., 1.], !!python/tuple [0.005, .995]]
    'model__alpha': [ .001, .01, .05, .03, .1 ]
  'ridge':
    'dropmissing__threshold': [ 0.9, 1 ]
    'droplowvariance__threshold': [ 0, 0.01 ]
    'winsorizer__limits': [!!python/tuple [0., 1.], !!python/tuple [0.005, .995]]
    'model__alpha': [ .001, .01, .05, .03, .1 ]
  'randomforest':
    'dropmissing__threshold': [ 0.9, 1 ]
    'droplowvariance__threshold': [ 0, 0.01 ]
    'winsorizer__limits': [!!python/tuple [0., 1.], !!python/tuple [0.005, .995]]
    'model__max_depth': [ 2, 4, 6, 8, 10 ]
    'model__n_estimators': [ 50, 100, 200 ]
  'gradientboosting':
    'dropmissing__threshold': [ 0.99 ]
    'droplowvariance__threshold': [ 0.01 ]
    'winsorizer__limits': [!!python/tuple [0., 1.], !!python/tuple [0.005, .995]]
    'model__min_data_in_leaf': [ 10, 20, 50 ]
    'model__num_leaves': [ 5, 10, 20 ]
    'model__learning_rate': [ 0.05, 0.075, 0.1 ]
    'model__n_estimators': [ 50, 100, 200 ]
```