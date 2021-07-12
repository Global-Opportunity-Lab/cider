from box import Box
from helpers.io_utils import load_model
from helpers.utils import *
from helpers.plot_utils import *
from helpers.ml_utils import *
import yaml


class Learner:

    def __init__(self, cfg_dir,
                 clean_folders=False, kfold=5):

        # Read config file
        with open(cfg_dir, "r") as ymlfile:
            cfg = Box(yaml.load(ymlfile, Loader=yaml.FullLoader))
        self.cfg = cfg

        # Prepare working directory
        self.features_fname = cfg.path.ml.features
        self.labels_fname = cfg.path.ml.labels
        self.outputs = cfg.path.ml.outputs
        make_dir(self.outputs, clean_folders)
        make_dir(self.outputs + '/untuned_models')
        make_dir(self.outputs + '/tuned_models')

        # Spark setup
        spark = get_spark_session(cfg)
        self.spark = spark

        # Load features
        self.features = self.spark.read.csv(self.features_fname, header=True)
        if 'name' not in self.features.columns:
            raise ValueError('Features dataframe must include name column')

        # Load labels
        self.labels = self.spark.read.csv(self.labels_fname, header=True)
        if 'name' not in self.labels.columns:
            raise ValueError('Labels dataframe must include name column')
        if 'label' not in self.labels.columns:
            raise ValueError('Labels dataframe must include label column')
        if 'weight' not in self.labels.columns:
            self.labels = self.labels.withColumn('weight', lit(1))
        self.labels = self.labels.select(['name', 'label', 'weight'])

        self.kfold = KFold(n_splits=kfold, shuffle=True, random_state=100)

        # Define models
        self.untuned_models = {
            'linear': Pipeline([('dropmissing', DropMissing(threshold=0.9)),
                                ('droplowvariance', VarianceThreshold(threshold=0.01)),
                                ('imputer', SimpleImputer(strategy='constant', fill_value=0)), 
                                ('winsorizer', Winsorizer(limits=(.005, .995))), 
                                ('scaler', StandardScaler()), 
                                ('model', LinearRegression())]),
            
            'lasso': Pipeline([('dropmissing', DropMissing(threshold=0.9)),
                                ('droplowvariance', VarianceThreshold(threshold=0.01)),
                                ('imputer', SimpleImputer(strategy='constant', fill_value=0)), 
                                ('winsorizer', Winsorizer(limits=(.005, .995))), 
                                ('scaler', StandardScaler()), 
                                ('model', Lasso(alpha=.05))]),

            'ridge': Pipeline([('dropmissing', DropMissing(threshold=0.9)),
                                ('droplowvariance', VarianceThreshold(threshold=0.01)),
                                ('imputer', SimpleImputer(strategy='constant', fill_value=0)), 
                                ('winsorizer', Winsorizer(limits=(.005, .995))), 
                                ('scaler', StandardScaler()), 
                                ('model', Ridge(alpha=.05))]),

            'randomforest': Pipeline([('dropmissing', DropMissing(threshold=0.9)),
                                        ('droplowvariance', VarianceThreshold(threshold=0.01)),
                                        ('imputer', SimpleImputer(strategy='constant', fill_value=0)), 
                                        ('winsorizer', Winsorizer(limits=(.005, .995))), 
                                        ('model', RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=1, max_depth=4))]),

            'gradientboosting': Pipeline([('dropmissing', DropMissing(threshold=0.9)),
                                            ('droplowvariance', VarianceThreshold(threshold=0.01)),
                                            ('winsorizer', Winsorizer(limits=(.005, .995))), 
                                            ('model', LGBMRegressor(n_estimators=100, n_jobs=-1, random_state=1, min_data_in_leaf=100, num_leaves=4, 
                                            learning_rate=0.1, verbose=-10))])
        }

        self.tuned_models = {
            'linear': Pipeline([('dropmissing', DropMissing()),
                                ('droplowvariance', VarianceThreshold()),
                                ('imputer', SimpleImputer(strategy='constant', fill_value=0)), 
                                ('winsorizer', Winsorizer()), 
                                ('scaler', StandardScaler()), 
                                ('model', LinearRegression())]),
            
            'lasso': Pipeline([('dropmissing', DropMissing()),
                                ('droplowvariance', VarianceThreshold()),
                                ('imputer', SimpleImputer(strategy='constant', fill_value=0)), 
                                ('winsorizer', Winsorizer()), 
                                ('scaler', StandardScaler()), 
                                ('model', Lasso())]),

            'ridge': Pipeline([('dropmissing', DropMissing()),
                                ('droplowvariance', VarianceThreshold()),
                                ('imputer', SimpleImputer(strategy='constant', fill_value=0)), 
                                ('winsorizer', Winsorizer()), 
                                ('scaler', StandardScaler()), 
                                ('model', Ridge())]),

            'randomforest': Pipeline([('dropmissing', DropMissing()),
                                        ('droplowvariance', VarianceThreshold()),
                                        ('imputer', SimpleImputer(strategy='constant', fill_value=0)), 
                                        ('winsorizer', Winsorizer()), 
                                        ('model', RandomForestRegressor(random_state=1, n_jobs=-1))]),

            'gradientboosting': Pipeline([('dropmissing', DropMissing()),
                                            ('droplowvariance', VarianceThreshold()),
                                            ('winsorizer', Winsorizer()), 
                                            ('model', LGBMRegressor(random_state=1, n_jobs=-1, verbose=-10))])
        }

        self.grids = cfg.hyperparams

    def merge(self):

        print('Number of observations with features: %i (%i unique)' % (self.features.count(), self.features.select('name').distinct().count()))
        print('Number of observations with labels: %i (%i unique)' % (self.labels.count(), self.labels.select('name').distinct().count()))

        merged = self.labels.join(self.features, on='name', how='inner')
        print('Number of matched observations: %i (%i unique)' % (merged.count(), merged.select('name').distinct().count()))

        save_df(merged, self.outputs + '/merged.csv')
        self.merged = pd.read_csv(self.outputs + '/merged.csv')
        self.x = self.merged.drop(['name', 'label', 'weight'], axis=1)
        self.y = self.merged['label']
        # Make the smallest weight 1
        self.weights = self.merged['weight']/self.merged['weight'].min()

    def untuned_model(self, model_name):

        make_dir(self.outputs + '/untuned_models/' + model_name)

        raw_scores = cross_validate(self.untuned_models[model_name], self.x, self.y,
                                    cv=self.kfold,
                                    return_train_score=True,
                                    scoring=['r2', 'neg_root_mean_squared_error'],
                                    fit_params={'model__sample_weight': self.weights})

        scores = {}
        scores['train_r2'] = '%.2f (%.2f)' % (raw_scores['train_r2'].mean(), raw_scores['train_r2'].std())
        scores['test_r2'] = '%.2f (%.2f)' % (raw_scores['test_r2'].mean(), raw_scores['test_r2'].std())
        scores['train_rmse'] = '%.2f (%.2f)' % (-raw_scores['train_neg_root_mean_squared_error'].mean(), -raw_scores['train_neg_root_mean_squared_error'].std())
        scores['test_rmse'] = '%.2f (%.2f)' % (-raw_scores['test_neg_root_mean_squared_error'].mean(), -raw_scores['test_neg_root_mean_squared_error'].std())
        with open(self.outputs + '/untuned_models/' + model_name + '/results.json', 'w') as f:
            json.dump(scores, f)

        # Save model
        model = self.untuned_models[model_name].fit(self.x, self.y, model__sample_weight=self.weights)
        dump(model, self.outputs + '/untuned_models/' + model_name + '/model')

        # Feature importances
        self.feature_importances(model_name=model_name, tuned=False)

        return scores

    def tuned_model(self, model_name):

        make_dir(self.outputs + '/tuned_models/' + model_name)

        model = GridSearchCV(estimator=self.tuned_models[model_name],
                             param_grid=self.grids[model_name],
                             cv=self.kfold,
                             verbose=0,
                             scoring=['r2', 'neg_root_mean_squared_error'],
                             return_train_score=True,
                             refit='r2',
                             n_jobs=-1)

        model.fit(self.x, self.y, model__sample_weight=self.weights)
        
        # Save tuning results
        tuning_results = pd.DataFrame(model.cv_results_)
        tuning_results.drop(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_fit_time'], axis=1)\
            .to_csv(self.outputs + '/tuned_models/' + model_name + '/tuning.csv', index=False)

        # Save accuracy results for best model
        best_model = tuning_results.iloc[tuning_results['mean_test_r2'].argmax()]
        scores = {}
        scores['train_r2'] = '%.2f (%.2f)' % (best_model['mean_train_r2'], best_model['std_train_r2'])
        scores['test_r2'] = '%.2f (%.2f)' % (best_model['mean_test_r2'], best_model['std_test_r2'])
        scores['train_rmse'] = '%.2f (%.2f)' % (-best_model['mean_train_neg_root_mean_squared_error'], -best_model['std_train_neg_root_mean_squared_error'])
        scores['test_rmse'] = '%.2f (%.2f)' % (-best_model['mean_test_neg_root_mean_squared_error'], -best_model['std_test_neg_root_mean_squared_error'])
        with open(self.outputs + '/tuned_models/' + model_name + '/results.json', 'w') as f:
            json.dump(scores, f)

        # Save model
        dump(model, self.outputs + '/tuned_models/' + model_name + '/model')

        # Feature importances
        self.feature_importances(model_name=model_name, tuned=True)

        return scores

    def feature_importances(self, model, tuned=True):
        subdir = '/tuned_models/' if tuned else '/untuned_models/'
        # Load model
        model_name, model = load_model(model, out_path=self.outputs, tuned=tuned)

        if 'feature_importances_' in dir(model.named_steps['model']):
            imports = model.named_steps['model'].feature_importances_
        else:
            imports = model.named_steps['model'].coef_

        imports = pd.DataFrame([self.x.columns, imports]).T
        imports.columns = ['Feature', 'Importance']
        imports = imports.sort_values('Importance', ascending=False)
        imports.to_csv(self.outputs + subdir + model_name + '/feature_importances.csv', index=False)
        return imports
    
    def oos_predictions(self, model, tuned=True):

        subdir = '/tuned_models/' if tuned else '/untuned_models/'
        model_name, model = load_model(model, out_path=self.outputs, tuned=tuned)

        oos = cross_val_predict(model, self.x, self.y, cv=self.kfold)
        oos = pd.DataFrame([list(self.merged['name']), list(self.y), oos]).T
        oos.columns = ['name', 'true', 'predicted']
        oos['weight'] = self.weights
        oos.to_csv(self.outputs + subdir + model_name + '/oos_predictions.csv', index=False)
        return oos

    def population_predictions(self, model, tuned=True, n_chunks=100):

        subdir = '/tuned_models/' if tuned else '/untuned_models/'
        model_name, model = load_model(model, out_path=self.outputs, tuned=tuned)
        
        columns = pd.read_csv(self.features_fname, nrows=1).columns

        chunksize = int(len(pd.read_csv(self.features_fname, usecols=['name']))/n_chunks)

        results = []
        for chunk in range(n_chunks):
            x = pd.read_csv(self.features_fname, skiprows=1 + chunk*chunksize, nrows=chunksize, header=None)
            x.columns = columns
            results_chunk = x[['name']].copy()
            results_chunk['predicted'] = model.predict(x[self.x.columns])
            results.append(results_chunk)
        results = pd.concat(results)

        results.to_csv(self.outputs + subdir + model_name + '/population_predictions.csv', index=False)
        return results

    def scatter_plot(self, model_name, tuned=True):

        subdir = '/tuned_models/' if tuned else '/untuned_models/'
        oos = pd.read_csv(self.outputs + subdir + model_name + '/oos_predictions.csv')
        oos['weight'] = 100*((oos['weight'] - oos['weight'].min())/(oos['weight'].max() - oos['weight'].min()))
        oos_repeat = pd.DataFrame(np.repeat(oos.values, oos['weight'], axis=0), columns=oos.columns).astype(oos.dtypes)
        corr = np.corrcoef(oos_repeat['true'], oos_repeat['predicted'])[0][1]

        grid = np.linspace(oos['true'].min(), oos['true'].max(), 100)
        l = loess(list(oos['true']), list(oos['predicted']))
        l.fit()
        pred = l.predict(grid, stderror=True)
        conf = pred.confidence()
        lowess = pred.values

        fig, ax = plt.subplots(1, figsize=(10, 7))

        ax.scatter(oos['true'], oos['predicted'], s=oos['weight'], label='Data', color='indianred')
        ax.plot(grid, lowess, color='mediumseagreen', label='LOESS Fit', linewidth=3)
        ax.fill_between(grid, conf.lower, conf.upper, color='mediumseagreen', alpha=0.1)

        ax.set_xlim(grid.min(), grid.max())
        ax.set_xlabel('True Value')
        ax.set_ylabel('Predicted Value')
        ax.set_title('True vs. Predicted Values (' + r'$\rho$' + ' = %.2f)' % corr)
        ax.legend(loc='best')
        clean_plot(ax)

        plt.savefig(self.outputs + subdir + model_name + '/scatterplot.png', dpi=300)
        plt.show()

    def feature_importances_plot(self, model_name, tuned=True, n_features=20):
        
        subdir = '/tuned_models/' if tuned else '/untuned_models/'
        importances = pd.read_csv(self.outputs + subdir + model_name + '/feature_importances.csv')

        importances = importances.sort_values('Importance', ascending=False)
        importances = importances[:n_features].sort_values('Importance', ascending=True)

        importances['color'] = importances['Feature']\
            .apply(lambda x: 'indianred' if x.split('_')[0] in ['cdr', 'international'] 
                else 'mediumseagreen' if x.split('_')[0] == 'location'
                else 'darkorange' if x.split('_')[0] == 'mobiledata'
                else 'dodgerblue' if x.split('_')[0] == 'mobilemoney'
                else 'orchid' if x.split('_')[0] == 'recharges'
                else 'grey')

        importances['Feature'] = importances['Feature']\
        .apply(lambda x: ' '.join(x.split('_')[1:])\
                .replace('percent', '%')\
                .replace('callandtext', '')\
                .replace('weekday', 'WD')\
                .replace('weekend', 'WE')\
                .replace('allday', '')\
                .replace('allweek', ''))

        fig, ax = plt.subplots(1, figsize=(20, 10))

        ax.barh(importances['Feature'], importances['Importance'], color=importances['color'])

        ax.set_title('Feature Importances', fontsize='large')
        ax.set_xlabel('Feature Importance')
        clean_plot(ax)

        plt.savefig(self.outputs + subdir + model_name + '/feature_importances.png', dpi=300)
        plt.show()

    def targeting_table(self, model_name, tuned=True):

        subdir = '/tuned_models/' if tuned else '/untuned_models/'
        oos = pd.read_csv(self.outputs + subdir + model_name + '/oos_predictions.csv')
        oos['weight'] = 100*(oos['weight'] - oos['weight'].min())/(oos['weight'].max() - oos['weight'].min())
        oos_repeat = pd.DataFrame(np.repeat(oos.values, oos['weight'], axis=0), columns=oos.columns).astype(oos.dtypes)

        grid = np.linspace(0, 90, 10)[1:]
        metric_grid = [metrics(oos_repeat['true'], oos_repeat['predicted'], p) for p in grid]

        table = pd.DataFrame()
        table['Proportion of Population Targeted'] = [('%i' % p) + '%' for p in grid]
        table['Pearson'] = np.corrcoef(oos_repeat['true'], oos_repeat['predicted'])[0][1]
        table['Spearman'] = spearmanr(oos_repeat['true'], oos_repeat['predicted'])[0]
        table['AUC'] = auc_overall(oos_repeat['true'], oos_repeat['predicted'])
        table['Accuracy'] = [('%i' % (g[0]*100)) + '%' for g in metric_grid]
        table['Precision'] = [('%i' % (g[1]*100)) + '%' for g in metric_grid]
        table['Recall'] = [('%i' % (g[2]*100)) + '%' for g in metric_grid]

        table = table.round(2)
        table.to_csv(self.outputs + subdir + model_name + '/targeting_table.csv', index=False)
        return table
