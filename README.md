# MOMA_replicate



## Project Goal

The purpose of this project is to implement an architecture partially described in the MOMA paper: Xie, Wanze, Siddharth Kapoor, Yi Liang, Michael Cooper, Juan Carlos Niebles, and Ehsan Adeli. “MOMA: Multi-Object Multi-Actor Activity Parsing.” Neural Information Processing Systems (2021). The goal is to recognize activities in videos. The paper takes advantage of hierarchical classifications of activities, sub-activities, and atomic actions. Detecting either of the latter can help detect the other correctly, ie: holding a pan is an atomic action of the sub-activity frying vegetables. 

## Description

I ended up replicating the video stream, leaving apart the graph stream. The model is capable of replicating the results (within a range of 5 %) of the standalone video stream through the X3D architecture on the MOMA dataset with a pretraining on the Kinetics-400 dataset: mAP (ours vs original) Activity 73,8% vs 70,9%, Sub-Activity 42,3% vs 40,6% and Atomic Actions 21,3% vs 27,1%.

## Conclusions

The difference between metrics mainly comes from the sampling strategy of the activities, sub-activities, and atomic actions in a video. My sampling and loss strategies are clearly focusing on activities and sub-activities at the cost of building a model efficient at recognizing atomic actions. 





