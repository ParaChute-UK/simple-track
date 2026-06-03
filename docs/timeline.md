# Timeline Output

As well as saving data to output, Simple-Track also returns the Timeline object that it uses for storing data. This can be useful if further processing or analysis is required on the tracking data.

## Loading the `Timeline`

The Timeline can be loaded in two ways:

1. Assigning to a variable if running from a python script:
    ```python
    timeline = Tracker(my_config).run(my_data)
    ```

2. Using the `LoadOutput` object to read previously saved data back into a Timeline object:
    ```python
    from simpletrack.frame_output import LoadOutput
    timeline = LoadOutput("path_to_simpletrack_data").load_to_timeline()
    ```

    Note that `LoadOutput` does not currently read back the raw data that was used to run the tracking, and therefore the `Frame.raw_field` attribute will remain None. This will also mean that any methods which use this attribute (e.g., `Frame.identify_features`) will not work unless the input data is loaded back into the Frame manually.

## Using the `Timeline`

The Timeline object is used as storage for all `Frame` objects that were created during tracking. These are stored in dicts, with the valid dt.datetime object used as the key. The complete dictionary can be retrieved using:

```python
all_frames = timeline.get_timeline()
```

Alternatively, individual `Frame` objects can be retrieved using a given datetime key by:

```python
requested_time = dt.datetime(...)
requested_frame = timeline.get_frame(requested_time)
```

The previous frame relative to the current time can also be retrieved using:

```python
previous_frame = timeline.get_previous_frame(requested_time)
```

## Using `Frames`

`Frames` contain all of the data associated with a given timestep. To check the time that a given `Frame` is valid for, use `Frame.time`.

A number of 2D fields of the same shape as the input data is stored in each `Frame`:

```python
feature_field = my_frame.feature_field # The ID of the feature at each pixel (0 is background)
lifetime_field = my_frame.lifetime_field # The age in timesteps of the feature at each pixel (0 is background)
y_flow, x_flow = my_frame.get_flow() # The y, x motion vectors that translated features to this timestep
```

Note that the flow fields for a given `Frame` are derived by comparing this `Frame` to the `Frame` valid at the closest previous timestep. Therefore, the motion vectors should be interpreted as the displacements that translated the features _to this timestep from the previous timestep_ rather than the displacements that translate these features _to the next timestep_. As such, the first `Frame` in a given `Timeline` will not have a valid flow field, and `frame.get_flow()` will return `None, None`

As well as 2D fields, each `Frame` also contains a dictionary of `Feature` objects which contain more information about each tracked feature. This dictionary uses the integer id as key and returns the corresponding `Feature` object. The full dictionary can be accessed using:

```python
all_features = my_frame.features
```

Alternatively, individual `Features` can be accessed using:

```python
my_feature = my_frame.get_feature(feature_id=12)
```

## Using `Features`

Each `Feature` contains information about its location, extent, and interaction with other `Features`. Each property can be accessed using `my_feature.{property_name}`

| Feature Property    | Description |
| -------- | ------- |
| `id` (int) | Unique identifier    |
| `time` (dt.datetime) | Time that the feature is valid  |
| `max` (float) | Maximum value of the input data across the feature |
| `mean` (float) | Mean value of the input data across the feature |
|  | __Spatial data__  |
| `size` (int) | Number of pixels spanned by the feature |
| `coords` (NDArray) | All (y, x) coordinates in the domain spanned by the feature |
| `centroid` (tuple) | Central (y, x) position of the feature |
| `major_vector` (tuple) | Unit (y, x) vector of feature semi-major axis |
| `minor_vector` (tuple) | Unit (y, x) vector of feature semi-minor axis |
| `major_radius` (float) | Radius of feature semi-major axis |
| `minor_radius` (float) | Radius of feature semi-minor axis |
| | __Tracking data__ |
| `lifetime` (int)   | The number of timesteps the feature has been tracked for  |
| `provisional_id` (int) | Used in `FrameTracker` to provisionally match IDs between different frames |
| `parent` (int) | ID of the parent feature that this feature split from, if applicable |
| `child` (list) | List of IDs that split from this feature, if applicable |
| `accreted` (list) | List of IDs that have been accreted by this feature, if applicable | 
| `accreted_in_next_frame_by`  (int) | ID of feature that accretes this feature in the next frame, if applicable
