import os
import shutil
from glob import glob


def copy_pcd(input_pcd_dir, output_pcd_dir):
    file_list = sorted(glob(input_pcd_dir))
    if not os.path.exists(output_pcd_dir):
        os.makedirs(output_pcd_dir)

    for i, source_file in enumerate(file_list):
        print(source_file)
        des_file = os.path.join(output_pcd_dir, f"{i + 1:06d}") + '.pcd'
        shutil.copy(source_file, des_file)


if __name__ == '__main__':
    # dataset path
    dataset_root = "/media/jlutripper/My Passport/v2x_dataset"
    test_ego_dir = os.path.join(dataset_root, "test/*/0/*.pcd")
    test_cp_dir = os.path.join(dataset_root, "test/*/1/*.pcd")

    des_ego_folder = os.path.join(dataset_root, "v2v_test/0/pcd")
    des_cp_folder = os.path.join(dataset_root, "v2v_test/1/pcd")

    # save with id (1-1993)
    copy_pcd(test_ego_dir, des_ego_folder)
    copy_pcd(test_cp_dir, des_cp_folder)
