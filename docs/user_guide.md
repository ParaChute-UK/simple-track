
# Command-Line Interface

`
simpletrack [-h] configs [configs ...] [-i, --path] [-l, --loader] [-ia, --iterate_over_array] [-dim, --iterating_dim]
`

* -h: Help command
* configs: path to one or more yaml config files
* -i, --path: Input path for finding data to load and track
* -l, --loader: Path to a loader file and function name, of the form `file.py|func_name`
* -ia, --iterate_over_array: If flagged, tells the code to load a single file and iterate over a dimension of the loaded data
* -dim, --iterating_dim: If -ia is flagged, tells the code which dimension to iterate over

# Input Requirements
While Simple-Track is designed to accept a wide range of input data, certain requirements must be met for the tool to function as intended:

* The input data must be gridded and contain a consistent spatial domain and resolution between frames.

* The features of interest must be defined by a threshold value, and these features must translate as a result of a spatially consistent background flow.

* The time between frames should be sufficiently short such that features can be reasonably expected to persist between frames. This is not a strict requirement since the tool includes an artificial advection step that projects data onto a common time, but it is likely that longer time steps will lead to more errors in feature matching and therefore less accurate tracking statistics. 

# Running Simple-Track

Simple-Track can be run in two ways:

## 1. Running Simple-Track from the Command Line
* Simple-Track can be run from the command line with a config file as an additional argument:

    ```
    simpletrack my_config.yaml 
    ```

* The `my_config.yaml` file contains the parameters for running Simple-Track. The required parameters are shown below:

    ```yaml
    INPUT:
        path: /path_to_folder_containing_data/*.data
        loader: /path_to_file_containing_function|function_name # See next section
    FEATURE:
        threshold: 1 # Threshold used for defining a feature
    ```
    
* Other parameters, such as `experiment_name`, `output_path` and `save_data`, along with more technical options, can also be set in this config file. See [All Simple-Track Parameters](#all-simple-track-parameters) for a full list.

* A valid loader function is required for pre-processing input data before tracking. See [Loading Data](#loading-data) for more information.

* Any number of config files can be provided as additional arguments, Simple-Track will iterate over each one in turn.

## 2. Importing Simple-Track to a python file
* Simple-Track can be run by importing the `Tracker` class from the `simpletrack` module. A config can be input either using a path to a yaml file, or by passing a dict when instantiating the object:

    ```python
    from simpletrack import Tracker

    my_config = {
        INPUT: {
            path: "/path_to_folder_containing_data/*.data",
            loader: "/path_to_file_containing_function|function_name" # See next section
        },
        FEATURE: {
            threshold: 1, # Threshold used for defining a feature
        }
    }

    timeline = Tracker(my_config).run()

    # Alternatively, if these parameters are saved in a config file, the path to this config can also be set as input
    timeline = Tracker("./my_config.yaml").run()
    ```
* Other parameters, such as `experiment_name`, `output_path` and `save_data`, along with more technical options, can also be set in this config. See [All Simple-Track Parameters](#all-simple-track-parameters) for a full list.

* If `loader` is included as a config input, the specified function is used for pre-processing input data before tracking. Alternatively, valid pre-processed data may be passed to the `Tracker.run()` method, bypassing the use of a separate function, and eliminating the need for the `INPUT` config section. See [Loading Data](#loading-data) for more information.

* `Tracker.run()` returns a `Timeline` object which is used to store all tracking and feature data. This can be inspected and analysed beyond the [outputs](#outputs) that are saved as part of standard operation.

# Loading Data

Each Simple-Track input must contain two sets of data:

1. A `datetime` object specifying the time that the data is valid for
2. A `numpy.array` object containing the data to track

There are three methods of providing these data pairs to Simple-Track:

## 1. Loading through config options
* Simple-Track will load all data matching the structure given in `"INPUT": "path"` config section. This input supports wildcard matching (i.e., using `"./path_to_data/*.data"` would load all files with the `.data` suffix). 

* Since Simple-Track is a data-agnostic tool, there may be any number of bespoke tools for loading and pre-processing data before it is suitable for tracking. This functionality can be contained in a custom loader function that will perform these actions before passing the compatible data to the main processing workflow.

* An example of a custom loader function is shown below:

    ```python
    def user_definable_load(self, filename):
        import iris # Import any required libraries here

        # Get 2D data from input file as a numpy array
        cube = iris.load_cube(filename, "precipitation_flux")
        data = cube.data

        # Additional data pre-processing can be performed here too!

        # Get time from input file, in datetime format
        tcoord = cube.coord("time")
        time = tcoord.units.num2pydate(tcoord.points)[0]

        # Method must return a tuple of 
        # (datetime.datetime, numpy.NDArray), where the 
        # first element is the time the data is valid for
        # and second element is the 2D array of data to be tracked
        return time, data
    ```

* This loader function is then specified in the `"INPUT": "loader"` config using the `./path_to_file.py|func_name` format. So in this case, the config option would be `./path_to_file.py|user_definable_load`.

* Loading via the config can be used whether Simple-Track is being run [from the command line](#1-running-simple-track-from-the-command-line) or [from a python file](#2-importing-simple-track-to-a-python-file). 

## 2. Loading through the Command Line
* The same `"INPUT"` config sections mentioned above can also be input [from the command line](#1-running-simple-track-from-the-command-line)

    ```
    simpletrack my_config.yaml -i /path_to_folder/*.data -l ./path_to_file.py|func_name
    ```

## 3. Passing a dict directly to Tracker.run()
* If SimpleTrack is being run [from a python file](#2-importing-simple-track-to-a-python-file) and a suitable set of data has already been loaded, this data can be passed directly to `Tracker.run()` as a `dict`, with the `datetime` object as the key and a `numpy.array` object as the value. For example:

    ```python
    import datetime as dt
    import numpy as np
    from simpletrack import Tracker

    time1 = dt.datetime(year=2000, month=1, day=1, hour=10, minute=5)
    time2 = time1 + dt.timedelta(minutes=5)

    data1 = np.array(...)
    data2 = np.array(...)

    st_input = {
        time1: data1,
        time2: data2,
    }

    my_config = {...}

    Tracker(my_config).run(st_input)
    ```

* Any number of time:data pairs can be passed to `Tracker.run()` and the code will iterate over the ordered dict.

* Passing data into `Tracker.run()` via this method will bypass any `"INPUT":"loader"` or `"INPUT":"path"` inputs specified in the corresponding config file.

# Extra Loading Parameters
The loader function inputs expect that each file contains data at a single time, meaning that each file will be loaded and tracked sequentially.

However, if data at multiple times are contained in a single file, this can be handled by Simple-Track using the following options.

### Iterate Over Arrays
* To tell Simple-Track to just load a single file and iterate over a given dimension of the loaded data, use the following config options:

    ```yaml
    INPUT:
        path: /path_to_folder_containing_data/*.data
        loader: /path_to_file_containing_function|function_name # See next section
        iterate_over_array: True
        iterating_dim: 0
    ```

* The `iterate_over_array` flag tells the code to just load and iterate over a single input

* The `iterarting_dim` argument tells the code which dimension to iterate over. 

* If this option is used, the loader function should also return a list of datetime.datetime objects of the same size as the iterating dimension. 

# Input/Output Config Parameters
A Simple-Track config is ordered into sections, each of which control different parts of the code. This part of the guide describes `INPUT` and `OUTPUT`. Explanations for the `FEATURE`, `FLOW_SOLVER` and `TRACKING` sections are [given in the workflow docs](workflow.md)

```yaml
INPUT:
  path: ./path_to_input_data/*.data
  loader: /path_to_file_containing_function|function_name
  iterate_over_array: False 
  iterating_dim: 0 
```
path (str):
* Only required if not passing data to `Tracker.run()` directly
* Path to the input data and data format to load
* Supports wildcard matching (i.e., `*.data` will load all files with the .data suffix)

loader (str):
* Only required if not passing data to `Tracker.run()` directly
* Path to the file containing the function to load data
* Also, separated by `|`, is the function name to use to load the data

iterate_over_array (bool, optional):
* Defaults to `False`
* To tell Simple-Track to just load a single file and iterate over a given dimension of the loaded data
* See [above](#iterate-over-arrays) for more

iterating_dim (int):
* Defaults to `0`
* Only required if `iterate_over_array` is set to `True`
* Sets the dimension of the array to iterate over

```yaml
OUTPUT:
  path: ./output
  experiment_name: Simple-Track Experiment
  save_data: true 
  skip_tracking: false
  output_raw_data: true  
```

path (str, optional):
* Defaults to `.\output`
* Path for saving data into

experiment_name (str, optional):
* Defaults to `Simple-Track Experiment`
* Name of the Simple-Track experiment to add to outputs

save_data (bool, optional):
* Defaults to `True`
* Flag to save the data to output

skip_tracking (bool, optional):
* Defaults to `False`
* If enabled, will only perform feature identification (and produce the relevant `Feature` and `Frame` objects), but will not attempt to match these features between inputs.

output_raw_data (bool, optional):
* Defaults to `True`
* When enabled, will save a copy of the input data to the simple-track output folder. This is then used when re-loading data into a Timeline using the LoadOutput object. This ensures loaded Timelines are functionally identical to freshly generated timelines. 
* If set to `False`, LoadOutput will only load data generated by Simple-Track. To make these timelines functionally identical to generated ones, the raw data will then need to be loaded separately into each Frame.

# Loading Previously Saved Data
Simple-Track includes functionality to load previously saved data back into a `Timeline` object for further analysis. See the example below:


```python
from simpletrack.frame_output import LoadOutput

path_to_saved_data = "/path/to/saved/data"
loader = LoadOutput(path_to_saved_data)
timeline = loader.load_to_timeline()
```

Here, the `path_to_saved_data` is the path containing the `.fields` and `.csv` files output from Simple-Track. Typically, this will be the same path specified in the config option `["OUTPUT"]["path"]`.

The `LoadOutput().load_to_timeline()` method will search the directory for all Simple-Track output files, and load the field and csv files into `Frame` and `Feature` objects respectively. 

**NOTE**: To ensure that a `Timeline` loaded using this method is functionally identical to a `Timeline` returned by a call to `Tracker.run()`, it is important to leave the `["OUTPUT"]["output_raw_data"]` as its default value of `True`. Doing so will tell Simple-Track to save a copy of the input data to the output directory, meaning it can then be read back in using just the path to this directory (which is the only input to `LoadOutput`). 

However, it may not always be desirable to save copies of this data, especially if tracking is being run on an extended period or each data instance is large. Setting `["OUTPUT"]["output_raw_data"]` to `False` will therefore skip this step, and will mean that `LoadOutput` does not load anything into the raw_field for each Frame. This is usually not too restrictive, and most of the existing functionality of these objects can still be used as expected. However, to re-gain full functionality, the input data will need to be loaded into the `raw_field` property of each respective `Frame`.  