# Mitigating Covariate Shift for Low Dimensional Machine Learning Problems via Lattice Based Models 
 
## About this project
Bachelor Thesis submitted to the School of Business and Economics of Humboldt-Universität zu Berlin for the degree B.Sc. Economics.

### Built With

* [R](https://www.r-project.org/)
* [Python](https://www.python.org/)
* [Tensorflow](https://www.tensorflow.org/)

## Abstract

## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

To run the code it is necessary to create a virtual environment for Python and TensorFlow. For doing so just run the following code:
* Python
  ```sh
  conda env create -f environment.yml
  ```
Alternatively `scr/03full_script.R` contains instructions how to set up a `virtualenv` environment.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Humboldt-WI/dissertations/covariate_shift.git
   ```
2. Go to project directory
   ```sh
   cd ~/path/to/project/directory/covariate_shift
   ```   
3. Install required R packages
   ```sh
   Rscript scr/helpers.R --install
   ```

## Preprocessing

To preprocess the data go to `scr` directory and run
   ```sh
   Rscript 01preprocess.R --dir --write=TRUE
   ```
This will preprocess the data and write it in `csv`format to `data/cleaned`. In case that there is no data available in `data/raw`, just run the above command with the `--web` option instead of `--dir`; this will automatically download all necessary data from corresponding web sources.

Furthermore there are additional options available to obtain EDA plots For more information see
   ```sh
   Rscript 01preprocess.R -h
   ```

## Training and evaluation

To train the model employed in the thesis run
   ```sh
   Rscript 02train.R --save
   ```
By default the model is only trained and evaluated on a single small data set and not on the entire data. To run the model on the full data, just suply the `-f` flag. The `--save` flag stores the training and test ROC values for the lattice and tree model in a data frame and writes it to `out/res`. Furthermore it is possible to change model parameters as batch size, epochs, etc. from the command line via corresponding flags and values. For default values call the help flag.

## References

 \[1\] Gupta, M., A. Cotter, J. Pfeifer, K. Voevodski, K. Canini, A. Mangylov, W. Moczydlowski, and A. Van Esbroeck (2016): “Monotonic calibrated interpolated look-up tables,” The Journal of Machine Learning Research, 17, 3790–3836.

\[2\] Garcia, E., R. Arora, and M. R. Gupta (2012): “Optimized regression for eﬀicient function evaluation,” IEEE Transactions on Image Processing, 21, 4128–4140.

\[3\] Garcia, E. and M. Gupta (2009): “Lattice regression,” Advances in Neural Information Processing Systems, 22, 594–602.

\[4\] Moreno-Torres, J. G., T. Raeder, R. Alaiz-RodríGuez, N. V. Chawla, and F. Herrera (2012): “A unifying view on dataset shift in classification,” Pattern recognition, 45, 521–530.

```
$ pwd
/path/to/project/directory/ba_vwl

$ ls
|- ba_vwl.Rproj
|- data/
   |- cleaned/
   |- raw/
	   |- german/
	   |- gmc/
	   |- pak/
	   |- taiwanese/
   |- README.md
|- environment.yml
|- out/
   |- res/
   |- plots/
|- README.md
|- scripts/
   |- feature_configs.py
   |- helpers.R
   |- tfl_experiment.R

``