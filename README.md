# Indoor 3D Egocentric Object (IDEO) Dataset

## Overview
IDEO is composed of 58K egocentric images of 6 object categories used in daily activities, collected by 85 subjects using RGB-D cameras. Each image is associated with precisely fitted multiple 3D objects with respect to 3D shape and 9-DOF pose (translation, rotation, scale, and aspect ratio). The dataset includes not only static objects but also dynamic objects manipulated by the subjects, which introduces nontrivial characteristic occlusion and uncommon object poses. The action labels for each activity in the video sequence is also annotated.  

## Download

IDEO dataset can be downloaded through the AWS CLI:

```
$ aws s3 cp s3://ideo-dataset/submission_public.zip /path/to/local/directory
```

