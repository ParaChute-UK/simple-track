# Why Choose Simple-Track?

Simple-Track provides a versatile, easy to use framework for tracking features across a wide range of datasets.


## Features, not Particles

Simple-Track focuses on tracking the evolution of extended objects (features) that are defined using easy-to-interpret thresholds. Any deformation, fragmentation, or unification of features is identified and recorded, which allows users to construct a complete interaction history for all feature of interest. This approach contrasts other "particle" trackers, which instead treat features as point-like objects that can only change through translation. 

Particle identification can also be more complicated than feature identification, with multiple pre- and post-tracking filters used to remove spurious particles or trajectories. These additional processing layers add complexity to the tracker design, which can reduce output interpretability. 

The standard Simple-Track configuration does not include these filtering steps. Instead, features in every frame are defined based on the same two input criteria: contiguous regions above a given size and threshold. By assuming inputs are already processed and ready to track, Simple-Track minimises the number of tunable parameters that can affect output consistency and reliability. 


## Specialised for Fluid Dynamics

Computer vision object trackers are designed to identify complicated motion fields from scenes where objects can overlap or translate with many possible scales of motion. This complexity is necessary for reliably tracking 3D objects projected onto an image plane. For fluids dynamics problems that aren't as sensitive to these parallax effects, this level of complexity is unnecessary. 

Simple-Track includes a built-in flow solver which assumes that each feature of interest changes from frame to frame primarily (though not exclusively) due to a physical background flow. This "flow dominance" is a core assumption behind the construction of Simple-Track, and facilitates a more efficient estimation of feature motion than other optical flow schemes. The background flow can be estimated directly from the data, and helps make tracking more robust by projecting features onto a common timestep before matching between frames. This flow field is also output by the code for further analysis. 

 
## Data Agnostic

This is perhaps the most important facet of Simple-Track. While the tool was originally developed as a method to track thunderstorms in models and observations, the code does not make any assumptions about the data it is being given. The only inputs required by the code are matched pairs of Numpy arrays and datetime objects. This makes the code suitable for use on any datasets with a consistent grid, and where features evolve primarily due to a physical background flow. 


## Powerful, Efficient, Modular
Efficiencies in Numpy and the bespoke flow solver mean each tracking step can be as quick as a few seconds. Extensive test coverage ensures the code remains accurate and robust throughout updates and new feature additions. User friendly data structures enable the same code that Simple-Track uses to store data for post-tracking analysis. All Simple-Track functionality is controllable through `yaml` config files, and can be run on the command line or within Python code. 

Additionally, the object-oriented framework and modular design makes it easy to change or replace components for particular purposes. For instance, Simple-Track includes a built-in flow solver that is accurate and efficient, but also includes two other commonly-used optical-flow schemes that users can switch between. Simple-Track can easily be extended to include other flow schemes, input processors, output formatters or tracking algorithms. 