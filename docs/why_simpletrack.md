# Why Choose Simple-Track?

Simple-Track provides a versatile, easy to use framework for tracking features across a wide range of datasets.


### Features, not Particles

Simple-Track focuses on tracking the evolution of extended objects (features) that are defined using easy-to-interpret thresholds. Any deformation, fragmentation, or unification of features is identified and recorded, which allows users to construct a complete interaction history for all feature of interest. This approach contrasts other "particle" trackers, which instead treat features as point-like objects that can only change through translation. 

Particle identification can also be more complicated than feature identification, with multiple pre- and post-tracking filters used to remove spurious particles or trajectories. These additional processing layers add complexity to the tracker design, which can reduce output interpretability. 

The standard Simple-Track configuration does not include these filtering steps. Instead, features in every frame are defined based on the same two input criterion: contiguous regions above a given size and threshold. By assuming inputs are already processed and ready to track, Simple-Track minimises the number of tunable parameters that can affect output consistency and reliability. 


### Specialised for Fluid Dynamics

It's not for computer vision purposes, so flow solver is faster since no assumption of parallax. Instead, it is assumed that each feature of interest changes from frame to frame primarily (though not exclusively) due to a background flow. This flow can be estimated directly from the data and is output by the code.

This flow field also helps ensure tracking is more robust by projecting features onto a common timestep before matching between frames. 
 
### Data Agnostic

Perhaps the most important part of Simple-Track. While this was originally developed as a method to track thunderstorms in models and observations, the code does not make any assumptions about the data it is being given. The only inputs required by the code is matched pairs of numpy arrays and datetime objects.

### Powerful and Efficient
Efficiencies in numpy and with bespoke flow solver mean each tracking step can be as quick as a few seconds, though this scales with the number of tracked features. Extensive test coverage ensures code remains accurate and robust throughout updates and new feature additions. User friendly data structures enable the same code that Simple-Track uses to store data for post-tracking analysis. 

### Modular 
Object oriented framework and modular design makes it easy to change or replace components for particular purposes. 