# Simple-Track: A Data-Agnostic, Flow-Dependent Python Object Tracker

## Summary
Simple-Track is a threshold-based object tracking algorithm for 2D data, designed to track the complex interactions that can emerge between objects whose primary motion is determined by a physical, background flow field. Under such fields, objects are tracked between consecutive frames by estimating their location at a common timeframe and matching based on the degree of overlap. Matched objects retain the same identification between all tracked frames, while new objects are assigned a unique label. Additionally, objects that translate under divergent flows may merge with each other, or split into multiple separate objects. Simple-Track compiles comprehensive information about feature merging, splitting, initiation and dissipation.

## State of the Field
More general object trackers, e.g. for computer vision, and object recognition. Image recognition purposes are slightly different than what's used here, since object motion is not usually linked to a background flow (some exceptions, e.g. traffic flow, although the flow field here is not physical.)

Acknowledgement of origin of this algorithm: geophysical sciences/meteorology and cloud tracking. 

Titan: multi-use radar processing suite which includes clutter rejection, quality control, cartesian projection and, most relevant, real-time storm tracking. Closest to Simple-Track, and in fact this takes most of its logic from this. However, key difference, no feature projection step. Instead, Titan uses assumptions about the maximum storm size and speed to constrain a path optimisation solution over all identified objects. Simple-Track matching, meanwhile, assumes the feature projection step will have aligned the closest matching object from the previous timestep, thereby allowing a simple overlap comparison with objects in the vicinity of the object being matched.




In met, desire for temporal linkage between objects as a richer comparison source for model evaluation. This facilitates more in-depth analysis, such as convective initiation which NWP models typically struggle with accurately representing (timing, intensity, distribution etc...)


Within geophysical sciences, Simple-Track largely inspired by TITAN code... There are other trackers too such as tobac, MODE, etc. Primary difference between these 

Mention Feng et al 2025: Tracker MIP


## Statement of Need
While ST origin is in met, ST designed to fill a niche, which can provide the foundation for other object tracking applications. Designed to be data-agnostic, with only expected input being matched key:value pairs of datetime objects and numpy arrays. 

## Algorithm Design

## User Interaction

## Research Impacts

### Weather and Climate Model Evaluation
Within NWP, difficulty with properly evaluating performance of models with higher resolution of precip etc. due to double penalty problem. Some approaches smooth model and obs fields. Others instead look at statistical properties of convective cells, instead asserting that models should attempt to match size and intensity distributions of obs. 

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