## Installation

Ensure that you have python and anaconda installed. Clone the github repo:

```
git clone https://github.com/emilylaiken/cider
```

Install dependencies in a virtual environment; activate the environment.

```
cd cider
conda env create -n cider --file env.yml
conda activate cider
```

If using jupyter notebooks (for example to run the unit test notebooks), run the following to install the virtual environment as a kernel. 

```
conda install ipykernel
python -m ipykernel install --user --name=cider --display_name="cider"
```