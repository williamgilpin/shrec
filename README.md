## shrec

The **sh**ared **rec**urrence method for identifying causal drivers of collections of time series 

![A diagram of the algorithm](shrec/data/fig_github.png)

## Usage

```python
from shrec.models import RecurrenceManifold

X = load_data() # shape (n_timepoints, n_series)
model = RecurrenceManifold()
y_recon = model.fit_predict(X)
```

Additional examples can be found in the [`demos.ipynb`](demos.ipynb) notebook.


## Installation

Install the latest development version directly from GitHub

	git clone https://github.com/williamgilpin/shrec
	cd shrec
	pip install -I . 
    
## Additional information

Install core dependencies

	conda install numpy scipy matplotlib networkx h5py numba

Install scanpy using conda. If you are using bash, you may need to remove the quotes around networkx

	conda install -c conda-forge scikit-learn scanpy python-igraph leidenalg 
	pip3 install --upgrade-strategy only-if-needed graspologic 'networkx[default]'

Install optional dependencies

	conda install -c conda-forge jupyterlab
	conda install seaborn pandas 

Dependencies
+ pandas
+ scikit-learn
+ scipy
+ numpy
+ h5py
+ scanpy
+ python-igraph
+ leidenalg
+ networkx
+ graspologic

Additional dependencies for certain demonstrations
+ matplotlib 
+ seaborn


## Development

We welcome any advice or improvments. Please feel free to reach reach out, raise issues, or submit pull requests!


