from src.preprocess import *


# Set data directory
DATA_DIR = '.\\data'

# Get full image path by adding filename to base path

# Get the paths
test_paths = np.array([os.path.relpath(i, DATA_DIR).split('.')[0] for i in glob.glob(f'{DATA_DIR}/test_dataset/**/*.jpg')])

print(f"Test length: {len(test_paths)}")

# Sort paths to make sorting by patient easier
test_paths.sort()

# Check specific index
idx = 10
print(test_paths[idx:idx+10])


# Combine duplicate masks
deal_with_duplicates(data_dir=DATA_DIR, paths=test_paths)

# Recreate list of test paths now duplicates have been removed
test_paths_dup_rem = np.array([os.path.relpath(i, DATA_DIR).split('.')[0] for i in glob.glob(f'{DATA_DIR}/test_dataset/**/*.jpg')])
test_paths_dup_rem.sort()

# Define list of patient ids
print(list(set([path for path in test_paths_dup_rem])))

test_patient_ids = list(set([get_patient_id(path) for path in test_paths_dup_rem]))

overlap_adjustments(data_dir=DATA_DIR, test_paths=test_paths_dup_rem, test_patient_ids=test_patient_ids)


# patient_id = test_patient_ids[10]
# print(f'PATIENT: {patient_id}')
# patient_paths = get_patient_paths(test_paths, patient_id)
# for path in patient_paths:
#     print(path)
#     plot_image_and_mask(path)