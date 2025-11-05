"Screen-cam Imitation Module for Improving Data Hiding Robustness"
Aleksandr Fedosov, Kristina Dzhanashia, and Oleg Evsutin.

The pre-trained model is located at https://drive.google.com/drive/folders/1brH3yZk2keE6QOdZo__uAGYNSEN6G8tH?usp=sharing.

**Usage instructions**

To run this program:
1. Set up a Python interpreter (virtual environment) using Python 3.9+.
2. Download all the files or clone them from GitHub and place them in the project directory.
3. Install dependencies from requirements.txt by running **pip install -r requirements.txt**.
4. Place the pre-trained model in the project directory.

**Data preparation**

All images for processing must be saved in **./images/cover/[subdirectories]** e.g. "./images/cover/firstbatch/1.jpg". The naming of subdirecties and individual images is irrelevant.
By default only .jpg images proccessed, the working format can be changes inside the code.
After processing the resulting images are saved in **./images/output/[subdirectories]**  e.g. "./images/output/firstbatch/1.jpg".

**Core interface descriptions**

The program consists of one .py file.

The main components include:

* **PairedImageDataset** – Custom PyTorch Dataset that loads paired “cover” and “final” images from a dataset directory for inference.
* **tensor_to_pil(tensor_img)** – converts normalized PyTorch tensors back to PIL images.
* **apply_basic_transform(image, ...) / apply_advanced_transform(image, ...)** – geometric and noise-based tranformations
* **HDRNet** – neural network model for screen-cam initation
* **STN (Spatial Transformer Network)** – optional module for spatial alignment or warping (not directly used in inference loop)
* **MultiScaleDiscriminator / NLayerDiscriminator** – discriminator networks

Workflow:

1. Loads paired input images from ./images/cover and (optionally) ./images/final, if reference image are available.
2. Loads **HDRNet** weights (checkpoint_hdrnet_v2_epoch_50.pth).
3. Runs inference on each image (or image pair) to generate simulation results.
4. Applies selected geometric trasnform (basic or advanced).
5. Saves output images to ./images/output preserving subfolder structure.
