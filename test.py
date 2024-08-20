from src.preprocess import *
import torch
from torch.utils.data import DataLoader, Dataset
from src.datasets import MycetomaDataset
from src.UNet2D import UNet2D
from src.metrics import batch_dice_coeff, bce_dice_loss, dice_coefficient
from src.postprocessing import threshold_mask, post_process_binary_mask
from src.utils import visualize_segmented_image
from tqdm import tqdm

###########################################################################
# Initialization
###########################################################################
# Set data directory
DATA_DIR = '.\\data'
Model_Weights_Path='.\\model_saves\\updated_and_augmented_best_model.pth' #updated_masks_longer2_best_model.pth'
Results_Dir='results'
# Get full image path by adding filename to base path


###########################################################################
# Import models and Testing
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
        
        post_proc_mask_img = Image.fromarray(post_proc_mask)
        #post_proc_mask_img.save(Results_Dir + '/masks/' + str(idx) + '_mask.tif')

        imagename_output=Results_Dir + '/output_images/' + str(idx) + '.jpg'
        visualize_segmented_image(post_proc_mask, im, pred, imagename_output)

        #TODO Give post_proc_mask to Ben model
        # Output type of grain
        # display classification result on image
        # metrics
        
        


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