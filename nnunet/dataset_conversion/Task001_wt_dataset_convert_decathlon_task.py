from collections import OrderedDict
from nnunet.paths import nnUNet_raw_data
from batchgenerators.utilities.file_and_folder_operations import *
import shutil

base = "/home/zhang/zxc/nnUNet/DATASET/raw_data/brain_data/"

task_id = 1
task_name = "WTBrainSegmentation"
prefix = 'ses'
foldername = "Task%02.0d_%s" % (task_id, task_name)

out_base = join(nnUNet_raw_data, foldername)
imagestr = join(out_base, "imagesTr")
imagests = join(out_base, "imagesTs")
labelstr = join(out_base, "labelsTr")
maybe_mkdir_p(imagestr)
maybe_mkdir_p(imagests)
maybe_mkdir_p(labelstr)

train_folder = join(base, "imagesTr")
label_folder = join(base, "labelsTr")
test_folder = join(base, "imagesTs")
train_patient_names = []
test_patient_names = []

train_patients = [i.split('/')[-1] for i in subdirs(train_folder)]
for p in train_patients:
    serial_number = int(p[4:])
    train_patient_name = f'{prefix}-{serial_number:06d}'
    image_file = join(train_folder, f"{p}/{p}.nii.gz")
    label_file = join(label_folder, f"{p}/{p}.nii.gz")
    if isfile(image_file) and isfile(label_file):
        shutil.copy(image_file, join(imagestr, f'{train_patient_name}.nii.gz'))
        shutil.copy(label_file, join(labelstr, f'{train_patient_name}.nii.gz'))
    else:
        if isfile(image_file):
            print(f"No such file or directory: {label_file}")
            continue
        else:
            print(f"No such file or directory: {image_file}")
            continue
    train_patient_names.append(train_patient_name)

test_patients = [i.split('/')[-1] for i in subdirs(test_folder)]
for p in test_patients:
    serial_number = int(p[4:])
    test_patient_name = f'{prefix}-{serial_number:06d}'
    image_file = join(test_folder, f"{p}/{p}.nii.gz")
    shutil.copy(image_file, join(imagests, f'{test_patient_name}_0000.nii.gz'))
    test_patient_names.append(test_patient_name)

json_dict = OrderedDict()
json_dict['name'] = "wt BRATS"
json_dict['description'] = "the datasets is from wt"
json_dict['tensorImageSize'] = "3D"
json_dict['release'] = "0.0"
json_dict['modality'] = {
    "0": "T2w",
}
json_dict['labels'] = OrderedDict({
    "0": "region0",
    "1": "region1",
    "2": "region2",
    "3": "region3",
    "4": "region4",
    "5": "region5",
    "6": "region6",
    "7": "region7",
    "8": "region8",
    "9": "region9"
})
json_dict['numTraining'] = len(train_patient_names)
json_dict['numTest'] = len(test_patient_names)
json_dict['training'] = [{'image':f"./imagesTr/{train_patient_name}.nii.gz", 'label':f"./labelsTr/{train_patient_name}.nii.gz"} for train_patient_name in train_patient_names]
json_dict['test'] = [f"./imagesTs/{test_patient_name}.nii.gz" for test_patient_name in test_patient_names]

save_json(json_dict, os.path.join(out_base, "dataset.json"))
