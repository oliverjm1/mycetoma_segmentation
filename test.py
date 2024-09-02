from src.preprocess import *
import torch
from torch.utils.data import DataLoader, Dataset
from src.datasets import MycetomaDataset
from src.UNet2D import UNet2D
from src.metrics import batch_dice_coeff, bce_dice_loss, dice_coefficient
from src.postprocessing import threshold_mask, post_process_binary_mask
from src.utils import visualize_segmented_image, visualize_image_classified_segmented
from src.utils import format_file_paths_simplified
from tqdm import tqdm
from monai.networks.nets import DenseNet169
from torchvision.utils import save_image
###########################################################################
# Initialization
###########################################################################
# Set data directory
DATA_DIR = '.\\data'
Testing_Labels_Available=False # it should be true if Challenge Owners provide labels for teh generation of metrics
Model_Weights_Path='.\\model_saves\\updated_and_augmented_best_model.pth' #updated_masks_longer2_best_model.pth'
Model_Weights_Classification_Path='.\\model_saves\\dense_net_pretrained_weights.pth'
Results_Dir='.\\results'
# Get full image path by adding filename to base path


# Get the paths
test_paths_orig = np.array([os.path.relpath(i, DATA_DIR).split('.')[0] for i in glob.glob(f'{DATA_DIR}/test_dataset/**/*.jpg')])

print(f"Test length: {len(test_paths_orig)}")

# Sort paths to make sorting by patient easier
test_paths_orig.sort()

# Check specific index
idx = 10
print(test_paths_orig[idx:idx+10])

###########################################################################
# Delete and combining duplicate masks
###########################################################################
# Combine duplicate masks
deal_with_duplicates(data_dir=DATA_DIR, paths=test_paths_orig)


###########################################################################
# Combine overlapping masks (by Image Similarity Prediction)
###########################################################################
# Recreate list of test paths now duplicates have been removed
test_paths_dup_rem = np.array([os.path.relpath(i, DATA_DIR).split('.')[0] for i in glob.glob(f'{DATA_DIR}/test_dataset/**/*.jpg')])
test_paths_dup_rem.sort()


# Define list of patient ids
print(list(set([path for path in test_paths_dup_rem])))

test_patient_ids = list(set([get_patient_id(path) for path in test_paths_dup_rem]))

overlap_adjustments(data_dir=DATA_DIR, test_paths=test_paths_dup_rem, test_patient_ids=test_patient_ids)



###########################################################################
# Import Segmentation models and Testing
###########################################################################
test_paths = np.array([os.path.relpath(i, DATA_DIR).split('.')[0] for i in glob.glob(f'{DATA_DIR}/test_dataset/**/*.jpg')])
test_dataset = MycetomaDataset(test_paths, DATA_DIR, test_flag=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Create and load model save
model = UNet2D(3, 1, 8)
state_dict = torch.load(Model_Weights_Path, map_location=torch.device(device))

# Sometimes, the model dictionary keys contain 'module.' prefix which we don't want
remove_prefix = True

if remove_prefix:
    remove_prefix = 'module.'
    state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()


# test each image
with torch.no_grad():
    for idx, (inputs) in enumerate(test_loader):    
        inputs = inputs.to(device)
        outputs = model(inputs)

        im = inputs[0].detach().cpu().permute(1,2,0).numpy()
        pred = threshold_mask(outputs[0][0].detach().cpu().numpy())

        # Post-process mask
        post_proc_mask = np.clip(post_process_binary_mask(pred, threshold_fraction=0.1), 0, 1)
        
        # save mask for classification stage
        post_proc_mask_img = Image.fromarray(post_proc_mask)
        post_proc_mask_img.save(Results_Dir + '/masks/' + str(idx) + '_mask.tif')

        # # save orig image for classification stage
        im_orig = inputs[0]
        save_image(im_orig, Results_Dir + '/input_images/' + str(idx) + '.jpg')

        # # # visualize mask
        imagename_output=Results_Dir + '/output_segmented_images/' + str(idx) + '.jpg'
        visualize_segmented_image(post_proc_mask, im, pred, imagename_output)

        #TODO Give post_proc_mask to Ben model
        # Output type of grain
        # display classification result on image
        # metrics
        
print("Segmentation Completed, Classification Started!!!")
###########################################################################
# Import Classification models and Testing
###########################################################################
# Create and load model save
model = DenseNet169(spatial_dims=2, in_channels=4, out_channels=1, pretrained=True)
state_dict = torch.load(Model_Weights_Classification_Path, map_location=torch.device(device))
test_seg_paths_for_classification = np.array(['.'.join(i.split('.'))  for i in glob.glob(f'{Results_Dir}\masks\*')])
test_img_paths_for_classification = np.array(['.'.join(i.split('.'))  for i in glob.glob(f'{Results_Dir}\input_images\*')])
print(f"Test length for Classification: {len(test_seg_paths_for_classification )}")
print(f"Test length for Classification: {len(test_img_paths_for_classification)}")
test_paths_classification = format_file_paths_simplified(test_seg_paths_for_classification , test_img_paths_for_classification)

print(f"Test length for Classification: {len(test_paths_classification)}")

test_dataset = MycetomaDataset(test_paths_classification, Results_Dir, classification_flag=True)
test_loader_classification = DataLoader(test_dataset, batch_size=1, shuffle=False)



model.eval()
threshold = 0.5

if not Testing_Labels_Available:
    with torch.no_grad():
        #for features, image_orig_input, mask_seg_predicted in test_loader_classification:
        for idx, (features, image_orig_input, mask_seg_predicted) in enumerate(test_loader_classification):  
            features = features.to(device)
            output = model(features)  # Remove batch dimension
            output = output.squeeze(1)

            # Store predictions and labels
            predicted_probs = torch.sigmoid(output)
            predicted_class = (predicted_probs >= threshold).type(torch.long)
            predicted_label=str(predicted_class.item())
            print(f"Predicted Label: {predicted_label}")
            if predicted_label =='0':
                predicted_class_name='BM'
            else:
                predicted_class_name='FM'

            # visualize detections
            image_orig_input = image_orig_input[0].detach().cpu()
            mask_seg_predicted = mask_seg_predicted[0].detach().cpu()
            imagename_output=Results_Dir + '/output_images/' + str(idx) + '.jpg' #Results_Dir + '/output_images/foo.jpg' 
            visualize_image_classified_segmented(image_orig_input, mask_seg_predicted,predicted_class_name, imagename_output)

           

###########################################################################
# Test
###########################################################################


###########################################################################
# Qualitative Results (Images with Segmentation and Classification)
###########################################################################

# for image in test_image:
#         plot_image(im, pred, gt)
#     # Plot prediction before and after processing
#         fig, ax = plt.subplots(1, 3, figsize=(10, 5))
#         ax[0].imshow(pred)
#         ax[0].set_title('Binary Mask')
#         ax[0].axis('off')

#         ax[1].imshow(post_proc_mask)
#         ax[1].set_title('Post-Proc Mask')
#         ax[1].axis('off')

#         ax[2].imshow(gt)
#         ax[2].set_title('GT')
#         ax[2].axis('off')

#         plt.show()
###########################################################################
# Evaluate Metrics
###########################################################################