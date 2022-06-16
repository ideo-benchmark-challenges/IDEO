import glob
import os
import pickle
import cv2
import numpy as np
import torch
from utils import get_camera_matrix, parse_intrinsics_file
from pytorch3d.io import load_obj


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

        # From raw_anno, build C_T_G
        R_np = anno['raw_anno']['r'][:3, :3]
        T_np = anno['raw_anno']['t']
        # scale the center accordingly
        T_np *= 1 / anno['raw_anno']['s'][0]
        C_T_G = np.eye(4)
        C_T_G[:3, :3] = R_np
        C_T_G[:3, 3] = T_np

        # build O_T_G (shifted-scaled version of G)
        O_T_G = np.eye(4)
        O_T_G[:3, :3] = anno['post_proc']['r']
        O_T_G[:3, 3] = anno['post_proc']['t']
        G_T_O = np.linalg.inv(O_T_G)

        # get C_T_O for rendering
        C_T_O = C_T_G @ G_T_O

        # scale the center accordingly
        C_T_O[:3, 3] *= anno['post_proc']['s'][0]

        R_np = C_T_O[:3, :3]
        T_np = C_T_O[:3, 3]

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

        # Render onto image
        # Load mesh
        mesh = load_obj(mesh_file, load_textures=False)
        v = mesh[0]
        f = mesh[1].verts_idx

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
    txt_file = '/home/kvuong/dgx-projects/mount/home/jingfanguo/Documents/workspace/NOCS_new/splits/idea_amt_anno-0606/knife/test.txt'
    video_root = '/home/kvuong/dgx-projects/mount/oitstorage/idea/submission_public'

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
