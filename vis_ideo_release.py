import glob
import os
import pickle
import cv2
import numpy as np
import torch
from utils import get_camera_matrix, parse_intrinsics_file
from pytorch3d.io import load_obj
import argparse


def convert_vert_to_camera_coords(v, K, R, t):
    """Convert the vertices in world coordinate to camera coordinates.

    Args:
        v: The vertices array.
        K: The camera matrix.
        R: The rotation matrix. (3 X 3 list)
        t: The translation matrix. (3 list)

    Outputs:
        v: The camera coordinate.
    """

    # Make the vertice upside down.

    # Convert R and t to numpy array.
    R = np.asarray(R)
    t = np.asarray(t)

    # Compute the mesh vertices positions on the image.
    Rt = np.eye(4)
    Rt[:3, 3] = t
    Rt[:3, :3] = R
    Rt = Rt[:3, :] # Convert 4 X 4 to 3 X 4
    P = np.ones((v.shape[0], v.shape[1]+1))
    P[:, :-1] = v
    P = P.T # 4 X N

    # # Pix3D notation
    K[0, 0] *= -1
    K[1, 1] *= -1

    # Project to image
    img_cor_points = np.dot(K, np.dot(Rt, P))
    img_cor_points = img_cor_points.T # N X 3

    img_cor_points[:, 0] /= img_cor_points[:, 2]
    img_cor_points[:, 1] /= img_cor_points[:, 2]

    # Convert the coordinates to pixel level.
    img_cor_points = img_cor_points.astype(int)

    return img_cor_points


def render_mesh(video_dir, fid):
    # Camera intrinsics
    intrinsics_txt_file = os.path.join(video_dir, 'intrinsics.txt')
    if os.path.isfile(intrinsics_txt_file):
        K = np.loadtxt(intrinsics_txt_file).astype(np.float32)
    else:
        intrinsics_json_file = os.path.join(video_dir, 'rgb_intrinsics.json')
        color_params = parse_intrinsics_file(intrinsics_json_file, 'FULL_OPENCV')
        K, _ = get_camera_matrix(color_params, 'FULL_OPENCV')
        # HACK Color images are resized by half to align with the depth images
        K[:2] *= 0.5

    # RGB image
    color_file = os.path.join(video_dir, 'color_undistorted', f'color_{fid:07d}.jpg')
    assert os.path.isfile(color_file)

    # PointRend segmentation masks
    pr_pkl_file = os.path.join(video_dir, 'point_rend', 'pkl', f'{fid:07d}.pkl')
    if not os.path.isfile(pr_pkl_file):
        print('{} not exists, skipping this frame'.format(pr_pkl_file))
        return
    with open(pr_pkl_file, 'rb') as f:
        pr = pickle.load(f)

    # Annotation files
    anno_dir = os.path.join(video_dir, 'annotations', f'{fid:07d}')
    anno_files = glob.glob(os.path.join(anno_dir, '*.pkl'))

    img = cv2.imread(color_file)

    # Overlay the annotated 3D object onto the RGB image
    for anno_file in anno_files:
        # print(anno_file)
        filename = os.path.basename(anno_file)
        label = filename.split('_')[0]
        point_rend_id = int(filename.split('_')[1].split('.')[0])
        with open(anno_file, 'rb') as f:
            anno = pickle.load(f)

        # Load mesh
        mesh_file = anno_file.replace('.pkl', '.obj')
        assert os.path.exists(mesh_file)
        mesh = load_obj(mesh_file, load_textures=False)
        v = mesh[0]
        f = mesh[1].verts_idx
        verts_np = v.numpy()

        '''
        There are 3 coordinate systems: camera frame (C), normalized object-centric frame (O), and global frame (G)
        
        From a raw mesh defined in the frame G, we center it at origin and normalize it to fit in unit cube. 
        This newly centered, normalized mesh is the one in .obj file that we provided. 
        Thus, this new mesh is defined in frame O, which is offset w.r.t. frame G by a translation (centering).
        We can write this as: O_v = s * (O_R_G @ G_v + O_t_G) (Eq. 1), with O_R_G = eye(3)
        These are defined in anno['post_proc'].
        
        On the other hand, we have anno['raw_anno'] which defines the scale and transformation of the raw mesh (defined in G) w.r.t. C, or C_T_G.
        We can write this as: C_v = s_raw * C_R_G @ G_v + C_t_G (Eq. 2)
        
        The following demo code simply recovers G_v from O_v using Eq. 1, then project onto the camera frame C using Eq. 2.
        '''
        # Transform from O (normalized + re-centered mesh) to G (original raw mesh)
        # O_v = s * (O_R_G @ G_v + O_t_G) => G_v = ...
        O_T_G = np.eye(4)
        O_T_G[:3, :3] = anno['post_proc']['r']
        O_T_G[:3, 3] = anno['post_proc']['t']
        G_T_O = np.linalg.inv(O_T_G)
        verts_np /= anno['post_proc']['s'][0]
        verts_np = verts_np @ G_T_O[:3, :3].T + G_T_O[:3, 3][None, :]

        # After getting G_v, use the raw annotation to project onto the image
        # C_v = s_raw * C_R_G @ G_v + C_t_G
        verts_np *= anno['raw_anno']['s'][0]  # scale the vertices by s_raw
        v = torch.from_numpy(verts_np)

        R_np = anno['raw_anno']['r'][:3, :3]
        T_np = anno['raw_anno']['t']

        R_pytorch3d = torch.tensor(R_np)[None, :].float()
        T_pytorch3d = torch.tensor(T_np)[None, :].float()
        R_pytorch3d = R_pytorch3d.clone().permute(0, 2, 1)
        R_pytorch3d[:, :, :2] *= -1
        T_pytorch3d[:, :2] *= -1

        # save the flipped one (the untransposed)
        R_out = R_pytorch3d[0].numpy().T
        T_out = T_pytorch3d[0].numpy()

        # Gotta transpose since PyTorch3d use RHS notion
        R_pytorch3d = torch.tensor(R_out.T)[None, :].float()
        t_pytorch3d = torch.tensor(T_out)[None, :].float()

        # Project to image, vertex-by-vertex
        # TODO: add PyTorch3D rendering code
        intrinsic_matrix = K.copy()
        v = convert_vert_to_camera_coords(v, intrinsic_matrix, R_out, T_out)

        # plot vertices
        # for point in v:
        #     cv2.circle(img, tuple(point[:2]), 2, (0, 0, 255), -1)

        # plot face lines
        for line in f:
            pt1 = tuple(v[line[0], :2])
            pt2 = tuple(v[line[1], :2])
            pt3 = tuple(v[line[2], :2])

            cv2.line(img, pt1, pt2, (0, 255, 0), 1)
            cv2.line(img, pt2, pt3, (0, 255, 0), 1)
            cv2.line(img, pt1, pt3, (0, 255, 0), 1)

    cv2.imshow('test img', img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt_split_file', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    args = parser.parse_args()

    txt_file = args.txt_split_file
    video_root = args.data_root

    # Read image paths and build the dataset dictionary
    with open(txt_file) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]

    data_dict = {}

    for scene in lines:
        splits = scene.split('_')
        scene_name = splits[0] + '_' + splits[1] + '_' + splits[2]

        if scene_name not in data_dict:
            data_dict[scene_name] = []

        data_dict[scene_name].append(int(splits[3]))

    # subsample by 5
    for scene_name in data_dict:
        data_dict[scene_name] = data_dict[scene_name][::5]

    # Main loop: iterate through frames and visualize
    for scene_name in data_dict:
        splits = scene_name.split('_')
        participant_id = splits[0] + '_' + splits[1]
        scene_id = splits[2]

        video_dir = os.path.join(video_root, participant_id, scene_id)
        assert os.path.exists(video_dir)
        frame_ids = data_dict[scene_name]

        for fid in frame_ids:
            render_mesh(video_dir, fid)
