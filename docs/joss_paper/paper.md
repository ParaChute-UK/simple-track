# Simple-Track: A Data-Agnostic, Flow-Dependent Python Object Tracker

## Summary
Simple-Track is a threshold-based object tracking algorithm for 2D data, designed to track the complex interactions that can emerge between objects whose primary motion is determined by a physical, background flow field. Here, objects are defined as contiguous data regions matching a threshold condition. These objects are tracked between consecutive frames by predicting their location at a common timeframe and matching based on the degree of overlap. Matched objects retain the same identification between all tracked frames, while new objects are assigned a unique label. Additionally, objects that translate under divergent flows may merge with each other, or split into multiple separate objects. Simple-Track compiles comprehensive information about feature merging, splitting, initiation and dissipation.

## State of the Field
Tracking the motion of objects through discrete snapshots has been a longstanding problem, particularly within video and computer-vision research. Here, multi-object trackers (MOTs) track bounding boxes surrounding rigid bodies such as cars, people, or animals, whose motion is primarily self-determined. Most trackers follow a "tracking-by-detection" procedure, which first identifies relevant objects in a frame, and then links these objects between frames to form trajectories [ref SORT, DeepSORT and ByteTrack]. The largest issue limiting accuracy in these schemes is the estimation of 3D motion using only image-plane projections. Objects that pass in front of other objects can confuse algorithms unless they use occlusion-handling techniques, such as persistence tracking [ref SORT], re-identification using previous track properties [ref DeepSORT], or matching occluded tracks with "blurry" objects that have larger identification uncertainty [ref ByteTrack]. [Tracking accuracy can also be affected by camera shake or other erratic movements. Optical flow schemes help to alleviate these complications by estimating the motion of each pixel between frames. These can be either dense or sparse... These schemes are also successful in accounting for parallax effects by using a hierarchy of tracking resolutions to accurately capture larger-scale foreground motion and smaller-scale background motion.]

[Particle trackers...]

Tobac uses the trackpy formulation to... 
Trackers designed to track extended objects differ from MOTs and particle trackers in that they must account for merging and splitting of objects. 

Indeed, ST started as a cloud tracker in the same way that tobac is. 

In physical sciences, Also, "particle" trackers... see trackpy. This is also different... tobac builds on trackpy, nice link to feature tracking section.

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