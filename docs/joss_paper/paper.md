bibliography: tracking.bib

# Simple-Track: A Data-Agnostic, Threshold-Based Python Object Tracker

## Summary
Simple-Track is an object tracking algorithm for 2D data, designed to track the complex interactions that can emerge between extended objects. Here, objects are defined as contiguous data regions matching a threshold condition. These objects are tracked between consecutive frames by predicting their location at a common timeframe and matching based on the degree of overlap. Matched objects retain the same identification between all tracked frames, while new objects are assigned a unique label. Simple-Track also includes custom logic for identifying merging and splitting events, and uses this to compile a comprehensive feature interaction history. Feature matching is aided by the inclusion of a bespoke flow solver, making it specialised in tracking object motions that are linked to a physical flow field. The data-agnostic philosophy of Simple-Track is highlighted by the modular workflow design and support for user-configurable data loading routines.


## State of the Field
Tracking the motion of objects through discrete snapshots has been a longstanding problem, particularly within video and computer-vision research. Here, multi-object trackers (MOTs) track bounding boxes surrounding rigid bodies such as cars, people, or animals, whose motion is primarily self-determined. The largest issue limiting tracking accuracy in MOTs is the estimation of 3D motion using only image-plane projections. Objects that pass in front of other objects can confuse algorithms unless they use occlusion-handling techniques, such as persistence tracking [@bewley_simple_2016], re-identification using previous track properties [@wojke_simple_2017], or separately matching occluded tracks with "blurry" objects that have larger identification uncertainty [@zhang_bytetrack_2022]. Additionally, tracking accuracy can be affected by camera shake or other erratic movements. These undesirable motions can be compensated for using optical flow schemes, which estimate all-pixel (dense) or sharp-feature (sparse) motion vectors [@le_besnerais_dense_2005 @hamprecht_duality_2007 @kroeger_fast_2016 @ayzel_optical_2019]. Parallax effects are captured by using a hierarchy of tracking resolutions to accurately capture larger-scale foreground motion and smaller-scale background motion. 

Most MOTs follow a "tracking-by-detection" procedure, which first identifies relevant objects in a frame, and then links these objects between frames to form trajectories [@bewley_simple_2016 @wojke_simple_2017 @zhang_bytetrack_2022]. Bounding boxes surrounding objects of interest can either be identified using single-shot detectors that process an image in a single iteration [@redmon_you_2016], or using slower two-stage detectors that refine the initial estimation [@bewley_simple_2016]. From there, linking objects between frames typically follows the same three-stage process: motion prediction, similarity estimation, and object assignment. Motion prediction is most often performed using a constant-velocity Kalman filter [@bewley_simple_2016, @wojke_simple_2017, @zhang_bytetrack_2022]. Object similarity between the predicted and actual object locations can be assessed by using a simple overlap comparison, potentially with occlusion adjustments. Finally, objects are typically matched using The Hungarian algorithm [@bewley_simple_2016 @wojke_simple_2017 @zhang_bytetrack_2022], an efficient method for assigning two sets of items by minimising an associated cost function (in this case, the overlap between each object in both frames).

MOTs can either be configured for real-time, high-frequency object tracking or for more robust analysis after the event. In research contexts, it is usually more desirable to choose accuracy over efficiency, especially since input data may need pre-processing or interpolation before it is ready for tracking. Additionally, the definition of an object does not usually have the same level of ambiguity or require the same level of analysis as in MOTs. There are, however, different classes of trackers which treat objects differently: particle trackers and extended object trackers. Particle trackers assume that each object is composed of a single "bright" pixel, and that other surrounding bright points in the data are just an artifact of instrument noise. [Much work (it's not really a lot of work, need to change intro to this sentence)] goes into the finding the exact pixel producing the signal, discarding extra information about shape or area. Extended-object trackers, meanwhile, encompasses object morphology as a core part of the tracking process.

While most particle and extended-object trackers also employ the same tracking-by-detection procedure, their methods for identifying objects differ. The Crocker-Grier algorithm is one of the earliest and most popular particle-tracking algorithms, designed to track the motion of colloidal droplets suspended within another liquid [@crocker_methods_1996]. Here, particles are identified as the local maxima in brightness among points in the 70th percentile of brightness across the image. This brightness-weighted approach is also used in more modern tracking packages [@sbalzarini_feature_2005 @allan_trackpy_2025], though others favour a wavelet decomposition or gaussian-fit strategy [@jaqaman_robust_2008]. Linking strategies also differ between particle trackers, with some choosing nearest-neighbours [@allan_trackpy_2025], while others solve a global optimisation problem using a similar Hungarian algorithm approach [@jaqaman_robust_2008]. The design approach for each of these trackers is largely driven by their domain and purpose. As a result, extensions that handle object merging and splitting have become more popular, particularly with trackers used in molecular biology [@sbalzarini_feature_2005 @jaqaman_robust_2008].

<!-- [Additionally, extended objects can be defined more confidently when [more data...], making simpler methods for linking objects, such as overlaps, more viable (is this true?)]. -->

Compared to particle trackers, there are an even wider variety of uses for extended-object trackers. As such, these trackers can differ substantially in their construction from MOTs and particle trackers. For example, trackers that permit object shape deformations usually also include merging and splitting treatments. In biological imaging, mitosis and cell absorption can be tracked directly... 

[ in coronal holes...]

[In weather and climate, extended-object trackers are used for a wide variety of research purposes. Ocean eddies, sea-ice floes, wildfire perimeters...]


[More recently, trackers have become an important part of NWP model developers toolkit. Flex met creds here. See below for some discussion to summarise/include.]

Acknowledgement of origin of this algorithm: geophysical sciences/meteorology and cloud tracking. Multiple needs here: for tracking objects between discrete timesteps, and also for model evaluation. 

Within NWP, difficulty with properly evaluating performance of models with higher resolution of precip etc. due to double penalty problem. Some approaches smooth model and obs fields. Others instead look at statistical properties of convective cells, instead asserting that models should attempt to match size and intensity distributions of obs. 



Titan: multi-use radar processing suite which includes clutter rejection, quality control, cartesian projection and, most relevant, real-time storm tracking. Closest to Simple-Track, and in fact this takes most of its logic from this. However, key difference, no feature projection step. Instead, Titan uses assumptions about the maximum storm size and speed to constrain a path optimisation solution over all identified objects. Simple-Track matching, meanwhile, assumes the feature projection step will have aligned the closest matching object from the previous timestep, thereby allowing a simple overlap comparison with objects in the vicinity of the object being matched.


tobac:

PyFLEXTRAKR:

(see PlFLEXTRAKR paper intro for more trackers to compare/contrast)

In met, desire for temporal linkage between objects as a richer comparison source for model evaluation. This facilitates more in-depth analysis, such as convective initiation which NWP models typically struggle with accurately representing (timing, intensity, distribution etc...)


Within geophysical sciences, Simple-Track largely inspired by TITAN code... There are other trackers too such as tobac, MODE, etc. Primary difference between these 

Summarise these tracker MIPs after introducing the other trackers and explaining differences
Feng et al 2025: Tracker MIP. 

Also Prein et al 2024 tracker comparison project


## Statement of Need
While ST origin is in met, ST designed to fill a niche, which can provide the foundation for other object tracking applications. Designed to be data-agnostic, with only expected input being matched key:value pairs of datetime objects and numpy arrays. 

## Algorithm Design
The primary assumption behind Simple-Track matching algorithm is that, to first order, object evolution is largely controlled by background flow. 


## User Interaction

## Basic Example

## Research Impacts

### Weather and Climate Model Evaluation


### Hazardous Weather Impact Attribution

### Space Weather

## Documentation
<!-- Need to setup readthedocs, just a basic API for now, can pretty it up over xmas -->

## Distribution

## Acknowledgements
<!-- Mark Muetzelfeldt for code review, Ieuan Higgs for suggestions,  -->

## AI Usage Disclosure
Copilot for writing some tests and some docstrings. Core structure of the code and implementation done by humans...

## References