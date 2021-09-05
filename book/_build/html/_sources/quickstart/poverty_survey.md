---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

## Poverty Outcome Generation from Survey Data
Set up the configuration file and load some survey data (see {doc}`standardized data formats <../data_formats/placeholder>` for file schema). 

```
config.yml

path:
    survey: '/path/to/survey/data/directory'
    outputs: '/path/to/directory/to/save/outputs/'
    file_names:
        survey: '/survey_file_name.csv'
    
    col_types:
        survey: 
            continuous: ['con1', 'con2', 'con3']
            categorical: ['cat1', 'cat2', 'cat3']
            binary: ['bin1', 'bin2', 'bin3']
```

```{code-cell} ipython3
outcome_generator = SurveyOutcomesGenerator('/path/to/config/file', clean_folders=True)
```

Calculate PCA asset index and proxy-means test (PMT). Only use binary and continuous columns in the asset index.

```{code-cell} ipython3
asset_index = outcomes_generator.asset_index(cols=['con1', 'con2', 'bin1', 'bin2'])
```

Select five components to be used in the proxy-means test using forward selection of predictors with a linear regression. Calculate a proxy-means test with these components and obtain out-of-sample PMT predictions for the training survey. 

```{code-cell} ipython3
selected_cols, scores = outcomes_generator\
                        .select_features('consumption',
                                         ['con1', 'con2', 'cat1', 'cat2', 'bin1', 'bin2'], 
                                         5, 
                                         method=LinearRegession())
pmt = outcomes_generator.fit_pmt('consumption', 
                                 selected_cols, 
                                 model_name='linear', 
                                 winsorize=False, 
                                 scale=True)
```

[OUT] r2 score: 0.56


Use the trained proxy-means test on another survey dataset. 

```{code-cell} ipython3
predictions = outcomes_generator.pretrained_pmt('/path/to/other/data.csv', selected_cols, 'linear')
```