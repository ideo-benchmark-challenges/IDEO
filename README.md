# Indoor 3D Egocentric Object (IDEO) Dataset

## Overview
IDEO is composed of 58K egocentric images of 6 object categories used in daily activities, collected by 85 subjects using RGB-D cameras. Each image is associated with precisely fitted multiple 3D objects with respect to 3D shape and 9-DOF pose (translation, rotation, scale, and aspect ratio). The dataset includes not only static objects but also dynamic objects manipulated by the subjects, which introduces nontrivial characteristic occlusion and uncommon object poses. The action labels for each activity in the video sequence is also annotated.  

## Download

IDEO dataset can be downloaded directly [here](https://ideo-dataset.s3.amazonaws.com/submission_public.zip), or through the AWS CLI:

```
$ aws s3 cp s3://ideo-dataset/submission_public.zip /path/to/local/directory
```

## Data Organization

There is a separate directory for each sequence, organized by `participantID` and `sceneID`. Within each sequence, in terms of raw data, we store the (undistorted) RGB images (`color_undistorted`), depth images (`depth`), their correspondences (`color_to_depth_correspondences.txt`), and camera intrinsics (`rgb_intrinsics.json`). We also store the instance segmentation masks using PointRend under `point_rend` and the object annotations under `annotations`. For each object instance, the annotation data includes its scale as well as the orientation and translation w.r.t. the camera coordinate system (`.pkl` files). Each instance is also associated with their corresponding mesh (`.obj` file), category label, and the 2D instance segmentation mask.

The general data hierarchy is described below:

```
participantID
├── sceneID
    ├── color_undistorted
    │   └── color_<frameID>.png
    ├── depth
    │   └── depth_<frameID>.png
    ├── color_to_depth_correspondences.txt
    ├── rgb_intrinsics.json
    ├── point_rend
    │   └── <frameID>.pkl
    ├── annotations
    │   └── <frameID>
    │        ├── <objLabel>_<objID>.obj
    │        └── <objLabel>_<objID>.pkl
```
