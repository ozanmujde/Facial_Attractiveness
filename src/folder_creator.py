import os
import shutil

# specify your source directory
src_dir = '/Users/ozan/PycharmProjects/DeepLearningHW/SCUT_FBP5500_downsampled/training'
create_dir = '/Users/ozan/PycharmProjects/DeepLearningHW/SCUT/training'
# get all the filenames in the source directory
filenames = os.listdir(src_dir)
# iterate over the filenames
for filename in filenames:
    score = filename.split('_')[0]

    # create a new directory path
    new_dir = os.path.join(create_dir, score)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    shutil.move(os.path.join(src_dir, filename), new_dir)