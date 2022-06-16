import os
import json

import cv2
import numpy as np
import open3d as o3d
import trimesh
import torch


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def parse_intrinsics_file(filename, camera_model):
    ''' Parse an intrinsics json file
    Argument
        filename    - Input json file
    Return
        params      - Camera parameters
    '''
    with open(filename, 'r') as f:
        d = json.load(f)
        fx = d['fx']
        fy = d['fy']
        cx = d['cx']
        cy = d['cy']
        k1 =  d['k1']
        k2 =  d['k2']
        k3 =  d['k3']
        k4 =  d['k4']
        k5 =  d['k5']
        k6 =  d['k6']
        p1 =  d['p1']
        p2 =  d['p2']
    if str.upper(camera_model) == 'SIMPLE_PINHOLE':
        return [fx, cx ,cy]
    elif str.upper(camera_model) == 'PINHOLE':
        return [fx, fy, cx, cy]
    elif str.upper(camera_model) in ('SIMPLE_RADIAL', 'SIMPLE_RADIAL_FISHEYE'):
        return [fx, cx, cy, k1]
    elif str.upper(camera_model) in ('RADIAL', 'RADIAL_FISHEYE'):
        return [fx, cx, cy, k1, k2]
    elif str.upper(camera_model) == 'OPENCV':
        return [fx, fy, cx, cy, k1, k2, p1, p2]
    elif str.upper(camera_model) == 'FULL_OPENCV':
        return [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6]
    elif str.upper(camera_model) == 'OPENCV_FISHEYE':
        return [fx, fy, cx, cy, k1, k2, k3, k4]
    else:
        raise ValueError('Unsupported camera model: ' + camera_model)


def get_camera_matrix(camera_params, camera_model):
    ''' Get camera matrix and distortion coefficients from camera parameters in COLMAP format
    Arguments
        camera_params   - Camera parameters in COLMAP format
        camera_model    - Camera model
    Return
        K               - [3, 3] Camera matrix
        dc              - [12,] Distortion coefficients
    '''
    camera_params = np.asarray(camera_params)
    K = np.zeros([3, 3])
    K[2, 2] = 1
    dc = np.zeros([12,])
    if str.upper(camera_model) == 'SIMPLE_PINHOLE':
        K[0, 0] = camera_params[0]
        K[1, 1] = camera_params[0]
        K[0, 2] = camera_params[1]
        K[1, 2] = camera_params[2]
    elif str.upper(camera_model) == 'PINHOLE':
        K[0, 0] = camera_params[0]
        K[1, 1] = camera_params[1]
        K[0, 2] = camera_params[2]
        K[1, 2] = camera_params[3]
    elif str.upper(camera_model) in ('SIMPLE_RADIAL', 'SIMPLE_RADIAL_FISHEYE'):
        K[0, 0] = camera_params[0]
        K[1, 1] = camera_params[0]
        K[0, 2] = camera_params[1]
        K[1, 2] = camera_params[2]
        dc[0] = camera_params[3]
    elif str.upper(camera_model) in ('RADIAL', 'RADIAL_FISHEYE'):
        K[0, 0] = camera_params[0]
        K[1, 1] = camera_params[0]
        K[0, 2] = camera_params[1]
        K[1, 2] = camera_params[2]
        dc[0:2] = camera_params[3:5]
    elif str.upper(camera_model) == 'OPENCV':
        K[0, 0] = camera_params[0]
        K[1, 1] = camera_params[1]
        K[0, 2] = camera_params[2]
        K[1, 2] = camera_params[3]
        dc[0:4] = camera_params[4:8]
    elif str.upper(camera_model) == 'FULL_OPENCV':
        K[0, 0] = camera_params[0]
        K[1, 1] = camera_params[1]
        K[0, 2] = camera_params[2]
        K[1, 2] = camera_params[3]
        dc[0:8] = camera_params[4:12]
    elif str.upper(camera_model) == 'OPENCV_FISHEYE':
        K[0, 0] = camera_params[0]
        K[1, 1] = camera_params[1]
        K[0, 2] = camera_params[2]
        K[1, 2] = camera_params[3]
        dc[0:2] = camera_params[4:6]
        dc[4:6] = camera_params[6:8]
    else:
        raise ValueError('Unsupported camera model: ' + camera_model)
    return K, dc


def image_to_world(points_image, K, dc=None, num_iters=5):
    ''' Transform the points from image plane to normalized plane
    Arguments
        points_image    - [n, 2] Points on the image plane
        K               - [3, 3] Camera matrix
        dc              - [12,] Distortion coefficients organized as [k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4]
        num_iters       - Number of iterations for compensating distortion
    Return
        points_world    - [n, 2] Points on the normalized plane
    '''
    n = points_image.shape[0]
    points_world = np.hstack([points_image, np.ones([n,1])]) @ np.linalg.inv(K).T

    x = points_world[:, 0]
    y = points_world[:, 1]
    if dc is not None:
        x0 = x.copy()
        y0 = y.copy()
        for i in range(num_iters):
            r2 = x**2 + y**2
            r4 = r2**2
            r6 = r2**3
            icdist = (1 + dc[5]*r2 + dc[6]*r4 + dc[7]*r6) / (1 + dc[0]*r2 + dc[1]*r4 + dc[4]*r6)
            delta_x = 2*dc[2]*x*y + dc[3]*(r2 + 2*x**2) + dc[8]*r2 + dc[9]*r4
            delta_y = dc[2]*(r2 + 2*y**2) + 2*dc[3]*x*y + dc[10]*r2 + dc[11]*r4
            x = (x0 - delta_x) * icdist
            y = (y0 - delta_y) * icdist
    
    points_world = np.hstack([x.reshape([-1,1]), y.reshape([-1,1])])
    return points_world


def world_to_image(points_world, K, dc=None):
    ''' Transform the points from normalized plane to image plane
    Arguments
        points_world    - [n, 2] Points on the normalized plane
        K               - [3, 3] Camera matrix
        dc              - [12,] Distortion coefficients organized as [k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4]
    Return
        points_image    - [n, 2] Points on the image plane
    '''
    x = points_world[:, 0]
    y = points_world[:, 1]
    if dc is not None:
        r2 = x**2 + y**2
        r4 = r2**2
        r6 = r2**3
        icdist = (1 + dc[0]*r2 + dc[1]*r4 + dc[4]*r6) / (1 + dc[5]*r2 + dc[6]*r4 + dc[7]*r6)
        xx = x*icdist + 2*dc[2]*x*y + dc[3]*(r2 + 2*x**2) + dc[8]*r2 + dc[9]*r4
        yy = y*icdist + dc[2]*(r2 + 2*y**2) + 2*dc[3]*x*y + dc[10]*r2 + dc[11]*r4
        x = xx
        y = yy
    pn2d = np.stack([x, y, np.ones_like(x)], axis=1)
    points_image = pn2d @ K.T
    points_image = points_image[:, :2]
    return points_image


def rgbd_to_ptcloud(depth, K, dc=None, Rt=None, image=None):
    ''' Convert an RGB-D image to a (colored) point cloud
    Arguments
        depth   - [h, w] Depth image
        K       - [3, 3] Camera matrix
        dc      - [12,] Distortion coefficients organized as [k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4]
        Rt      - [4, 4] Camera extrinsics. When Rt is specified, the point cloud will be transformed 
                    to the world coordinates. Otherwise, point cloud will be in the camera coordinates
        image   - [h, w, 3] Color image. If unspecified, no color will be attached to the point cloud
    Return
        pcd     - An open3d.geometry.PointCloud object
    '''
    height = depth.shape[0]
    width = depth.shape[1]
    
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
    p2d = np.hstack([x_grid.reshape([-1,1]), y_grid.reshape([-1,1])])

    mask_valid = (depth.ravel() > 0)
    num_valid_points = mask_valid.sum()

    p2d = p2d[mask_valid, :]
    pn2d = image_to_world(p2d, K, dc)
    points = np.hstack([pn2d, np.ones([num_valid_points,1])]) * np.reshape(depth, [-1,1])[mask_valid]
    
    if Rt is not None:
        points = np.hstack([points, np.ones([num_valid_points,1])])
        points = points @ np.linalg.inv(Rt).T
        points = points[:, :3]

    pcd = o3d.geometry.PointCloud(o3d.cuda.pybind.utility.Vector3dVector(points))
    if image is not None:
        colors = np.reshape(image, [-1,3])[mask_valid,:]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd


# def transform(points, solution, center=None):
#     # points is an open3d.geometry.PointCloud object representing the point cloud being transformed
#     # solution is the 9x1 solution vector representing the 9dof transformation
#     scale = solution['s']
#     rotation = solution['r']
#     translation = solution['t']

#     # scale the vertices
#     try:
#         points.points = o3d.utility.Vector3dVector(
#             np.asarray(points.points) * np.array([scale]))
#     except:
#         points.vertices = o3d.utility.Vector3dVector(
#             np.asarray(points.vertices) * np.array([scale]))

#     # rotate
#     if center is None:
#         center = points.get_center()
#     points.rotate(rotation, center=center)

#     # translate
#     points.translate(translation)
#     return


def transform(points, solution, center=None):
    # points is an open3d.geometry.PointCloud object representing the point cloud being transformed
    # solution is the 9x1 solution vector representing the 9dof transformation
    scale = solution['s']
    rotation = solution['r']
    translation = solution['t']

    if center is None:
        center = points.get_center()
    
    try:
        v = np.asarray(points.points)
        v = scale * ((v - center[None,:]) @ rotation.T + center[None,:]) + translation[None,:]
        points.points = o3d.utility.Vector3dVector(v)
    except:
        v = np.asarray(points.vertices)
        v = scale * ((v - center[None,:]) @ rotation.T + center[None,:]) + translation[None,:]
        points.vertices = o3d.utility.Vector3dVector(v)

    
def draw_box(image_, box, label, kp_color):
    l, r, t, b = box[0], box[2], box[1], box[3]
    image = image_.copy()
    for c in range(3):
        image[t-2:t+2, l-2:r+2, c] = kp_color[c]
        image[b-2:b+2, l-2:r+2, c] = kp_color[c]
        image[t:b, l-2:l+2, c] = kp_color[c]
        image[t:b, r-2:r+2, c] = kp_color[c]

    text_pos = (l+1, t+10)
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    text_w, text_h = text_size
    cv2.rectangle(image, (l+1, t), (l+1 + text_w, t + text_h), np.zeros(3), -1)

    image = cv2.putText(image,
                        label,
                        text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,  # font size
                        255*np.ones(3),
                        0, cv2.LINE_AA)
    return image


def draw_mask(image_, mask, color):
    mask = mask[:, :, np.newaxis]
    color_mask = color * np.concatenate((mask, mask, mask), axis=2)

    image = cv2.addWeighted(image_, 1.0, np.asarray(color_mask, dtype=np.uint8), 0.3, 0)

    return image



def reorder_bbox_verts(bbox):
    ''' Reorder bbox vertices to make it compitable with pytorch3d.ops.box3d_overlap()
         pytorch3d           open3d
        [[0, 0, 0]         [[0, 0, 0]
         [1, 0, 0]          [1, 0, 0]
         [1, 1, 0]          [0, 0, 1]
         [0, 1, 0]   <--    [1, 0, 1]
         [0, 0, 1]          [0, 1, 0]          
         [1, 0, 1]          [1, 1, 0]          
         [1, 1, 1]          [0, 1, 1]          
         [0, 1, 1]          [1, 1, 1]]]
        See https://pytorch3d.readthedocs.io/en/latest/modules/ops.html#pytorch3d.ops.box3d_overlap
        bbox    -   Generated by o3d.geometry.TriangleMesh.create_box()
    '''
    idx_mapping = np.asarray([
        [0, 0],
        [1, 1],
        [2, 5],
        [3, 4],
        [4, 2],
        [5, 3],
        [6, 7],
        [7, 6],
    ], dtype=int)
    v = np.asarray(bbox.vertices)
    new_v = v[idx_mapping[:,1]]
    f = np.asarray(bbox.triangles)
    new_f = np.empty_like(f)
    for trg_vid, src_vid in idx_mapping:
        new_f[f == src_vid] = trg_vid

    new_bbox = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(new_v),
        o3d.utility.Vector3iVector(new_f)
    )
    if bbox.has_vertex_colors():
        vc = np.asarray(bbox.vertex_colors)
        new_vc = vc[idx_mapping[:,1]]
        new_bbox.vertex_colors = o3d.utility.Vector3dVector(new_vc)
    if bbox.has_vertex_normals():
        vn = np.asarray(bbox.vertex_normals)
        new_vn = vn[idx_mapping[:,1]]
        new_bbox.vertex_normals = o3d.utility.Vector3dVector(new_vn)
    if bbox.has_textures():
        new_bbox.textures = bbox.textures
    if bbox.has_triangle_material_ids():
        new_bbox.triangle_material_ids = bbox.triangle_material_ids
    if bbox.has_triangle_uvs():
        new_bbox.triangle_uvs = bbox.triangle_uvs

    return new_bbox
