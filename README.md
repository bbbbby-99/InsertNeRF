# InsertNeRF: Instilling Generalizability into NeRF with HyperNet Modules
## Installation
### Setup
Clone this repository:
  ```js
  git clone https://github.com/bbbbby-99/InsertNeRF.git
  cd InsertNeRF
  pip install -r requirements.txt
  ```
### Train generalization model
Train the model with NeuRay initialized from estimated depth of COLMAP:
  ```js
  python run_training.py --cfg configs/train/gen/insert_gen_depth_train.yaml
  ```
Models will be saved at data/model. 

### Render with trained models
  ```js
  python render.py --cfg configs/gen/insert_gen_depth_train.yaml
  ```
### Evaluation
Evaluation on all scenes in datasets, psnr/ssim/lpips will be printed on screen.
  ```js
  python eval.py
  ```

## Results
### InsertNeRF
Results for all scenes are obtained through our InsertNeRF rendering following Setting I, without any retraining in testing scenes.
<table>
  <tr>
    <td><img src="https://github.com/bbbbby-99/InsertNeRF/blob/main/gif%26image/lego.gif" alt="Lego GIF" width="200" /></td>
    <td><img src="https://github.com/bbbbby-99/InsertNeRF/blob/main/gif%26image/chair.gif" alt="Chair GIF" width="200" /></td>
    <td><img src="https://github.com/bbbbby-99/InsertNeRF/blob/main/gif%26image/mic.gif" alt="Ficus GIF" width="200" /></td>
    <td><img src="https://github.com/bbbbby-99/InsertNeRF/blob/main/gif%26image/ficus.gif" alt="Mic GIF" width="200" /></td>
  </tr>
  <tr>
    <td><img src="https://github.com/bbbbby-99/InsertNeRF/blob/main/gif%26image/hotdog.gif" alt="hotdog GIF" width="200" /></td>
    <td><img src="https://github.com/bbbbby-99/InsertNeRF/blob/main/gif%26image/ship.gif" alt="ship GIF" width="200" /></td>
    <td><img src="https://github.com/bbbbby-99/InsertNeRF/blob/main/gif%26image/drums.gif" alt="drums GIF" width="200" /></td>
    <td><img src="https://github.com/bbbbby-99/InsertNeRF/blob/main/gif%26image/materials.gif" alt="materials GIF" width="200" /></td>
  </tr>
</table>

<table>
  <tr>
    <td><img src="https://github.com/bbbbby-99/InsertNeRF/blob/main/gif%26image/hornsrgb.gif" alt="hornsrgb GIF" width="200" /></td>
    <td><img src="https://github.com/bbbbby-99/InsertNeRF/blob/main/gif%26image/fern.gif" alt="fernrgb GIF" width="200" /></td>
    <td><img src="https://github.com/bbbbby-99/InsertNeRF/blob/main/gif%26image/orchidsrgb.gif" alt="orchidsrgb GIF" width="200" /></td>
    <td><img src="https://github.com/bbbbby-99/InsertNeRF/blob/main/gif%26image/leavesrgb.gif" alt="leavesrgb GIF" width="200" /></td>
  </tr>
  <tr>
    <td><img src="https://github.com/bbbbby-99/InsertNeRF/blob/main/gif%26image/hornsdepth.gif" alt="hornsdepth GIF" width="200" /></td>
    <td><img src="https://github.com/bbbbby-99/InsertNeRF/blob/main/gif%26image/fern_depth.gif" alt="ferndepth GIF" width="200" /></td>
    <td><img src="https://github.com/bbbbby-99/InsertNeRF/blob/main/gif%26image/orchidsdepth.gif" alt="orchidsdepth GIF" width="200" /></td>
    <td><img src="https://github.com/bbbbby-99/InsertNeRF/blob/main/gif%26image/leavesdepth.gif" alt="leavesdepth GIF" width="200" /></td>
  </tr>
</table>

<table>
  <tr>
    <td><img src="https://github.com/bbbbby-99/InsertNeRF/blob/main/gif%26image/birdsrgb.gif" alt="birdsrgb GIF" width="200" /></td>
    <td><img src="https://github.com/bbbbby-99/InsertNeRF/blob/main/gif%26image/bricksrgb.gif" alt="bricksrgb GIF" width="200" /></td>
    <td><img src="https://github.com/bbbbby-99/InsertNeRF/blob/main/gif%26image/snowmanrgb.gif" alt="snowmanrgb GIF" width="200" /></td>
    <td><img src="https://github.com/bbbbby-99/InsertNeRF/blob/main/gif%26image/toolsrgb.gif" alt="toolsrgb GIF" width="200" /></td>
  </tr>
</table>

### Insert-NeRF++
<table>
  <tr>
    <td><img src="https://github.com/bbbbby-99/InsertNeRF/blob/main/gif%26image/Truckrgb.gif" alt="Truckrgb GIF" width="200" /></td>
    <td><img src="https://github.com/bbbbby-99/InsertNeRF/blob/main/gif%26image/Trainrgb.gif" alt="Trainrgb GIF" width="200" /></td>
    <td><img src="https://github.com/bbbbby-99/InsertNeRF/blob/main/gif%26image/Playgroundrgb.gif" alt="Playgroundrgb GIF" width="200" /></td>
    <td><img src="https://github.com/bbbbby-99/InsertNeRF/blob/main/gif%26image/M60rgb.gif" alt="M60rgb GIF" width="200" /></td>
  </tr>
</table>

### Insert-mip-NeRF
It will be released soon.
## Acknowledgements
In this repository, we build our codes based on the [NeuRay](https://github.com/liuyuan-pal/NeuRay). We thank all the authors for sharing great codes.

