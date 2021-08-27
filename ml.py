from autogluon.tabular import TabularPredictor
import autosklearn.regression
from box import Box
from helpers.io_utils import load_model
from helpers.utils import *
from helpers.plot_utils import *
from helpers.ml_utils import *
from datastore import *
import yaml


class Learner:

    def __init__(self, datastore: DataStore, clean_folders=False, kfold=5):
        self.cfg = datastore.cfg
        self.ds = datastore
        self.outputs = datastore.outputs + 'ml/'

        # Prepare working directories
        make_dir(self.outputs, clean_folders)
        make_dir(self.outputs + '/outputs/')
        make_dir(self.outputs + '/maps/')
        make_dir(self.outputs + '/tables/')

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
                                      ('model', RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=1,
                                                                      max_depth=4))]),

            'gradientboosting': Pipeline([('dropmissing', DropMissing(threshold=0.9)),
                                          ('droplowvariance', VarianceThreshold(threshold=0.01)),
                                          ('winsorizer', Winsorizer(limits=(.005, .995))),
                                          ('model', LGBMRegressor(n_estimators=100, n_jobs=-1, random_state=1,
                                                                  min_data_in_leaf=100, num_leaves=4,
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

        self.grids = self.ds.cfg.hyperparams

        # Load data into datastore
        data_type_map = {DataType.FEATURES: None,
                         DataType.LABELS: None}
        self.ds.load_data(data_type_map=data_type_map)
        self.ds.merge()

    def untuned_model(self, model_name):

        make_dir(self.outputs + '/untuned_models/' + model_name)

        raw_scores = cross_validate(self.untuned_models[model_name], self.ds.x, self.ds.y,
                                    cv=self.kfold,
                                    return_train_score=True,
                                    scoring=['r2', 'neg_root_mean_squared_error'],
                                    fit_params={'model__sample_weight': self.ds.weights})

        scores = {'train_r2': '%.2f (%.2f)' % (raw_scores['train_r2'].mean(), raw_scores['train_r2'].std()),
                  'test_r2': '%.2f (%.2f)' % (raw_scores['test_r2'].mean(), raw_scores['test_r2'].std()),
                  'train_rmse': '%.2f (%.2f)' % (-raw_scores['train_neg_root_mean_squared_error'].mean(),
                                                 -raw_scores['train_neg_root_mean_squared_error'].std()),
                  'test_rmse': '%.2f (%.2f)' % (-raw_scores['test_neg_root_mean_squared_error'].mean(),
                                                -raw_scores['test_neg_root_mean_squared_error'].std())}
        with open(self.outputs + '/untuned_models/' + model_name + '/results.json', 'w') as f:
            json.dump(scores, f)

        # Save model
        model = self.untuned_models[model_name].fit(self.ds.x, self.ds.y, model__sample_weight=self.ds.weights)
        dump(model, self.outputs + '/untuned_models/' + model_name + '/model')

        # Feature importances
        self.feature_importances(model=model_name, kind='untuned')

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

        model.fit(self.ds.x, self.ds.y, model__sample_weight=self.ds.weights)

        # Save tuning results
        tuning_results = pd.DataFrame(model.cv_results_)
        tuning_results.drop(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_fit_time'], axis=1) \
            .to_csv(self.outputs + '/tuned_models/' + model_name + '/tuning.csv', index=False)

        # Save accuracy results for best model
        best_model = tuning_results.iloc[tuning_results['mean_test_r2'].argmax()]
        scores = {'train_r2': '%.2f (%.2f)' % (best_model['mean_train_r2'], best_model['std_train_r2']),
                  'test_r2': '%.2f (%.2f)' % (best_model['mean_test_r2'], best_model['std_test_r2']),
                  'train_rmse': '%.2f (%.2f)' % (
                      -best_model['mean_train_neg_root_mean_squared_error'],
                      -best_model['std_train_neg_root_mean_squared_error']), 'test_rmse': '%.2f (%.2f)' % (
                -best_model['mean_test_neg_root_mean_squared_error'],
                -best_model['std_test_neg_root_mean_squared_error'])}
        with open(self.outputs + '/tuned_models/' + model_name + '/results.json', 'w') as f:
            json.dump(scores, f)

        # Save model
        dump(model, self.outputs + '/tuned_models/' + model_name + '/model')

        # Feature importances
        self.feature_importances(model=model_name, kind='tuned')

        return scores

    def automl(self, model_name):
        # Make sure model_name is correct, get relevant cfg
        assert model_name in ['autosklearn', 'autogluon']
        make_dir(self.outputs + '/automl_models/' + model_name)

        if model_name == 'autosklearn':
            cfg = self.cfg.params.automl.autosklearn
            model = autosklearn.regression.AutoSklearnRegressor(
                time_left_for_this_task=cfg.time_left,
                per_run_time_limit=None,
                ensemble_nbest=1,
                initial_configurations_via_metalearning=0,
                resampling_strategy=KFold,
                resampling_strategy_arguments={'n_splits': 5, 'shuffle': True, 'random_state': 100},
                n_jobs=cfg.n_jobs,
                memory_limit=cfg.memory_limit,
                seed=100)

            model.fit(self.ds.x, self.ds.y)
            model.refit(self.ds.x.copy(), self.ds.y.copy())
            dump(model.get_models_with_weights()[0][1], self.outputs + '/automl_models/' + model_name + '/model')

        elif model_name == 'autogluon':
            cfg = self.cfg.params.automl.autogluon
            train_data = pd.concat([self.ds.x, self.ds.y, self.ds.weights], axis=1)
            model = TabularPredictor(label=cfg.label,
                                     eval_metric=cfg.eval_metric,
                                     sample_weight=cfg.sample_weight,
                                     weight_evaluation=True,
                                     path=self.outputs + '/automl_models/' + model_name + '/model')
            model.fit(train_data,
                      presets='best_quality',
                      auto_stack=True,
                      time_limit=cfg.time_limit,
                      excluded_model_types=['FASTAI'])

        print('Finished automl training!')

    def feature_importances(self, model, kind='tuned'):
        # Load model
        subdir = '/' + kind + '_models/'
        model_name, model = load_model(model, out_path=self.outputs, type=kind)

        if 'feature_importances_' in dir(model.named_steps['model']):
            imports = model.named_steps['model'].feature_importances_
        else:
            imports = model.named_steps['model'].coef_

        imports = pd.DataFrame([self.ds.x.columns, imports]).T
        imports.columns = ['Feature', 'Importance']
        imports = imports.sort_values('Importance', ascending=False)
        imports.to_csv(self.outputs + subdir + model_name + '/feature_importances.csv', index=False)
        return imports

    def oos_predictions(self, model, kind='tuned'):
        # Load model
        subdir = '/' + kind + '_models/'
        model_name, model = load_model(model, out_path=self.outputs, type=kind)

        if model_name == 'autogluon':
            oos = model.get_oof_pred()
        else:
            oos = cross_val_predict(model, self.ds.x, self.ds.y, cv=self.kfold)
        oos = pd.DataFrame([list(self.ds.merged['name']), list(self.ds.y), oos]).T
        oos.columns = ['name', 'true', 'predicted']
        oos['weight'] = self.ds.weights
        oos.to_csv(self.outputs + subdir + model_name + '/oos_predictions.csv', index=False)
        return oos

    def population_predictions(self, model, kind='tuned', n_chunks=100):
        # Load model
        subdir = '/' + kind + '_models/'
        model_name, model = load_model(model, out_path=self.outputs, type=kind)

        columns = pd.read_csv(self.cfg.path.features, nrows=1).columns

        chunksize = int(len(pd.read_csv(self.cfg.path.features, usecols=['name'])) / n_chunks)

        results = []
        for chunk in range(n_chunks):
            x = pd.read_csv(self.cfg.path.features, skiprows=1 + chunk * chunksize, nrows=chunksize, header=None)
            x.columns = columns
            results_chunk = x[['name']].copy()
            results_chunk['predicted'] = model.predict(x[self.ds.x.columns])
            results.append(results_chunk)
        results = pd.concat(results)

        results.to_csv(self.outputs + subdir + model_name + '/population_predictions.csv', index=False)
        return results

    def scatter_plot(self, model_name, kind='tuned'):

        subdir = '/' + kind + '_models/'
        oos = pd.read_csv(self.outputs + subdir + model_name + '/oos_predictions.csv')
        oos['weight'] = 100 * ((oos['weight'] - oos['weight'].min()) / (oos['weight'].max() - oos['weight'].min()))
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

        importances['color'] = importances['Feature'] \
            .apply(lambda x: 'indianred' if x.split('_')[0] in ['cdr', 'international']
                   else 'mediumseagreen' if x.split('_')[0] == 'location'
                   else 'darkorange' if x.split('_')[0] == 'mobiledata'
                   else 'dodgerblue' if x.split('_')[0] == 'mobilemoney'
                   else 'orchid' if x.split('_')[0] == 'recharges'
                   else 'grey')

        importances['Feature'] = importances['Feature'] \
            .apply(lambda x: ' '.join(x.split('_')[1:])
                   .replace('percent', '%')
                   .replace('callandtext', '')
                   .replace('weekday', 'WD')
                   .replace('weekend', 'WE')
                   .replace('allday', '')
                   .replace('allweek', ''))

        fig, ax = plt.subplots(1, figsize=(20, 10))

        ax.barh(importances['Feature'], importances['Importance'], color=importances['color'])

        ax.set_title('Feature Importances', fontsize='large')
        ax.set_xlabel('Feature Importance')
        clean_plot(ax)

        plt.savefig(self.outputs + subdir + model_name + '/feature_importances.png', dpi=300)
        plt.show()

    def targeting_table(self, model_name, kind='tuned'):

        subdir = '/' + kind + '_models/'
        oos = pd.read_csv(self.outputs + subdir + model_name + '/oos_predictions.csv')
        oos['weight'] = 100 * (oos['weight'] - oos['weight'].min()) / (oos['weight'].max() - oos['weight'].min())
        oos_repeat = pd.DataFrame(np.repeat(oos.values, oos['weight'], axis=0), columns=oos.columns).astype(oos.dtypes)

        grid = np.linspace(0, 90, 10)[1:]
        metric_grid = [metrics(oos_repeat['true'], oos_repeat['predicted'], p) for p in grid]

        table = pd.DataFrame()
        table['Proportion of Population Targeted'] = [('%i' % p) + '%' for p in grid]
        table['Pearson'] = np.corrcoef(oos_repeat['true'], oos_repeat['predicted'])[0][1]
        table['Spearman'] = spearmanr(oos_repeat['true'], oos_repeat['predicted'])[0]
        table['AUC'] = auc_overall(oos_repeat['true'], oos_repeat['predicted'])
        table['Accuracy'] = [('%i' % (g[0] * 100)) + '%' for g in metric_grid]
        table['Precision'] = [('%i' % (g[1] * 100)) + '%' for g in metric_grid]
        table['Recall'] = [('%i' % (g[2] * 100)) + '%' for g in metric_grid]

        table = table.round(2)
        table.to_csv(self.outputs + subdir + model_name + '/targeting_table.csv', index=False)
        return table
