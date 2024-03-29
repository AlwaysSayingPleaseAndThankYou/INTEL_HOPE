from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import trimesh
from trimesh import Trimesh

# ========================================= #
# Set Path here
# EXAMPLE:
#   ROOT = '/path/to/FHAD/hand_pose_action'

ROOT = None
# ========================================= #


if ROOT is None:
    raise ValueError("Set the ROOT var to the path to the First-Person Hand Action Benchmark (F-PHAB)."
                     "Dataset can be found here: https://guiggh.github.io/publications/first-person-hands/")


# Loading utilities
def load_objects(obj_root_: Path) -> Dict[str, Trimesh]:
    """
    Load Polygon files for objects and convert them to Trimeshs
    """
    object_names = ['juice', 'liquid_soap', 'milk', 'salt']
    all_models = {}
    for obj_name in object_names:
        obj_path = Path(obj_root_, f'{obj_name}_model', f'{obj_name}_model.ply')
        mesh_ = trimesh.load(obj_path)
        all_models[obj_name] = mesh_

    # Hack to remedy mismatch with naming in data
    all_models['juice_bottle'] = all_models['juice']
    del all_models['juice']

    return all_models


def get_skeleton(sample_info: 'Sample', skel_root: Path) -> np.ndarray:

    skeleton_path = Path(skel_root,
                         sample_info.subject,
                         sample_info.action_name,
                         sample_info.seq_idx,
                         'skeleton.txt')
    # print(f'Loading skeleton from {skeleton_path}')
    skeleton_vals = np.loadtxt(skeleton_path)
    skeleton = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], 21, -1)[sample_info.frame_idx]
    return skeleton


def get_obj_transform(sample_info: 'Sample', obj_root_: Path) -> np.ndarray:
    """
    Load object transforms
    """
    seq_path = Path(obj_root_,
                    sample_info.subject,
                    sample_info.action_name,
                    sample_info.seq_idx,
                    'object_pose.txt')
    with open(seq_path) as seq_f:
        raw_lines = seq_f.readlines()
    raw_line = raw_lines[sample_info.frame_idx]
    line = raw_line.strip().split(' ')
    trans_matrix = np.array(line[1:]).astype(np.float32)
    trans_matrix = trans_matrix.reshape(4, 4).transpose()
    return trans_matrix


@dataclass
class Sample:
    subject: str
    action_name: str
    seq_idx: str
    frame_idx: int
    object_name: str


def main():
    # Setup Paths
    root_path = Path(ROOT)
    skeleton_root_path = Path(root_path, 'Hand_pose_annotation_v1')
    object_models_root_path = Path(root_path, 'Object_models')
    object_translations_root_path = Path(root_path, 'Object_6D_pose_annotation_v1_1')
    video_files_root_path = Path(root_path, 'Video_files')

    # Load object mesh
    object_infos = load_objects(object_models_root_path)
    reorder_idx = np.array([0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20])

    cam_extr = np.array([[0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
                         [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
                         [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
                         [0, 0, 0, 1]])

    cam_intr = np.array([[1395.749023, 0, 935.732544],
                         [0, 1395.749268, 540.681030],
                         [0, 0, 1]])

    # train
    images_train = []
    points2d_train = []
    points3d_train = []

    # val
    images_val = []
    points2d_val = []
    points3d_val = []

    # test
    images_test = []
    points2d_test = []
    points3d_test = []

    # Traverse the dataset directory and prepare the data for processing
    for subject in sorted(object_translations_root_path.iterdir()):
        if subject.name.startswith("."):
            continue
        print(f"* {subject}")

        for action_name in sorted(subject.iterdir()):
            if action_name.name.startswith("."):
                continue
            print(f' ∟ {action_name.name}')

            for seq_idx in sorted(action_name.iterdir()):
                if seq_idx.name.startswith("."):
                    continue
                print(f'   ∟ {seq_idx.name}')

                video_frame_dir = Path(video_files_root_path, *seq_idx.parts[-3:], 'color')
                if not video_frame_dir.exists():
                    print(f"WARNING: Could not find directory at {video_frame_dir}.  No video data is loaded.")
                    continue

                for each_frame in video_frame_dir.iterdir():
                    if each_frame.name.startswith("."):
                        continue

                    # Extract the filename's frame index to an int
                    #   i.e. 'color_0007.jpeg' -> 7
                    frame_idx = int(Path(each_frame).stem.split("_")[1])

                    # Remove the Action from object name
                    #  i.e 'pour_milk' -> 'milk'
                    obj = '_'.join(action_name.name.split('_')[1:])

                    sample = Sample(subject.name,
                                    action_name.name,
                                    seq_idx.name,
                                    frame_idx,
                                    obj
                                    )

                    # Load skeleton
                    skel = get_skeleton(sample, skeleton_root_path)[reorder_idx]

                    # Load object transform
                    obj_trans = get_obj_transform(sample, object_translations_root_path)

                    mesh = object_infos[sample.object_name]
                    verts = np.array(mesh.bounding_box_oriented.vertices) * 1000

                    # Apply transform to object
                    hom_verts = np.concatenate([verts, np.ones([verts.shape[0], 1])], axis=1)
                    verts_trans = obj_trans.dot(hom_verts.T).T

                    # Apply camera extrinsic to object
                    verts_camcoords = cam_extr.dot(verts_trans.transpose()).transpose()[:, :3]
                    # Project and object skeleton using camera intrinsics
                    verts_hom2d = np.array(cam_intr).dot(verts_camcoords.transpose()).transpose()
                    verts_proj = (verts_hom2d / verts_hom2d[:, 2:])[:, :2]

                    # Apply camera extrinsic to hand skeleton
                    skel_hom = np.concatenate([skel, np.ones([skel.shape[0], 1])], 1)
                    skel_camcoords = cam_extr.dot(skel_hom.transpose()).transpose()[:, :3].astype(np.float32)

                    skel_hom2d = np.array(cam_intr).dot(skel_camcoords.transpose()).transpose()
                    skel_proj = (skel_hom2d / skel_hom2d[:, 2:])[:, :2]

                    points = np.concatenate((skel_camcoords, verts_camcoords))
                    projected_points = np.concatenate((skel_proj, verts_proj))

                    if seq_idx.name == '1':  # val
                        images_val.append(str(each_frame))
                        points2d_val.append(projected_points)
                        points3d_val.append(points)
                    elif seq_idx.name == '3':  # test
                        images_test.append(str(each_frame))
                        points2d_test.append(projected_points)
                        points3d_test.append(points)
                    else:  # train
                        images_train.append(str(each_frame))
                        points2d_train.append(projected_points)
                        points3d_train.append(points)

    print("#=====Summary=====#")
    print(f"images_val count: {len(images_val)}")
    print(f"points2d_val count: {len(points2d_val)}")
    print(f"points3d_val count: {len(points3d_val)}")
    print()
    print(f"images_train count: {len(images_train)}")
    print(f"points2d_train count: {len(points2d_train)}")
    print(f"points3d_train count: {len(points3d_train)}")
    print()
    print(f"images_test count: {len(images_test)}")
    print(f"points2d_test count: {len(points2d_test)}")
    print(f"points3d_test count: {len(points3d_test)}")
    print()

    np.save('./images-train.npy', np.array(images_train))
    np.save('./points2d-train.npy', np.array(points2d_train))
    np.save('./points3d-train.npy', np.array(points3d_train))

    np.save('./images-val.npy', np.array(images_val))
    np.save('./points2d-val.npy', np.array(points2d_val))
    np.save('./points3d-val.npy', np.array(points3d_val))

    np.save('./images-test.npy', np.array(images_test))
    np.save('./points2d-test.npy', np.array(points2d_test))
    np.save('./points3d-test.npy', np.array(points3d_test))


if __name__ == "__main__":
    main()
