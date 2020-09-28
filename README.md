# **ARM**: Amodal 3D Reconstruction for Robotic Manipulation via Stability and Connectivity

<p align="center"><img src="https://github.com/wagnew3/Amodal-3D-Reconstruction-for-Robotic-Manipulationvia-Stability-and-Connectivity/blob/CORL_2020_Code_Release/webpage_assets/recon_comparison_video.gif" width="1000" /></p>

Learning-based 3D object reconstruction enables single- or few-shot estimation of 3D object models. For robotics, this holds the potential to allow model-based methods to rapidly adapt to novel objects and scenes. Existing 3D reconstruction techniques optimize for visual reconstruction fidelity, typically measured by chamfer distance or voxel IOU. We find that when applied to realistic, cluttered robotics environments, these systems produce reconstructions with low physical realism, resulting in poor task performance when used for model-based control. We propose ARM, an amodal 3D reconstruction system that introduces (1) a stability prior over object shapes, (2) a connectivity prior, and (3) a multi-channel input representation that allows for reasoning over relationships between groups of objects. By using these priors over the physical properties of objects, our system improves reconstruction quality not just by standard visual metrics, but also performance of model-based control on a variety of robotics manipulation tasks in challenging, cluttered environments.

<p align="center"><img src="https://github.com/wagnew3/Amodal-3D-Reconstruction-for-Robotic-Manipulationvia-Stability-and-Connectivity/blob/CORL_2020_Code_Release/webpage_assets/System.png" width="900" /></p>

The ARM framework creates a physics engine-based simulation from an RGBD image in four steps, summarized in the above figure:
1) We first apply an instance segmentation network to the input RGB-D image. 
2) For each object we detect, we pre-process its point cloud to compute its four channel voxel input representation, defined below.
3) ARM uses this representation to perform 3D shape estimation with a deep network, followed by post-processing.
4) We use representations obtain for manipulation planning. 

<p align="center"><img src="https://github.com/wagnew3/Amodal-3D-Reconstruction-for-Robotic-Manipulationvia-Stability-and-Connectivity/blob/CORL_2020_Code_Release/webpage_assets/loss_figure.png" width="800" /></p>

Optimizing for visual loss metrics like Chamfer distance along often results in reconstructions with poor physical fidelity during the planning phase, frequently due to instability of poor reconstruction of occluded regions.  We tackle this issue by designing auxiliary differentiable loss functions based on two physical priors: 1) objects are stable prior to manipulation, and 2) objects are a single connected component. The above figure gives an overview of these loss functions.

<p align="center">
  <img src="https://github.com/wagnew3/Amodal-3D-Reconstruction-for-Robotic-Manipulationvia-Stability-and-Connectivity/blob/CORL_2020_Code_Release/webpage_assets/occlusion_vs_cd.png" width="400" /><img src="https://github.com/wagnew3/Amodal-3D-Reconstruction-for-Robotic-Manipulationvia-Stability-and-Connectivity/blob/CORL_2020_Code_Release/webpage_assets/robot_manipulation_success_rates_vs_occlusion.png" width="400" /><br>
Left: reconstruction quality vs. occlusion, right: robot task success rate vs. occlusion
</p>
<br>

We generate a high quality synthetic cluttered tabletop dataset. We find that ARM reduces reconstruction Chamfer distance by 28% over the baseline, and even more for highly occluded objects. To evaluate the effectiveness of our reconstructions when they are used to create a simulation for robot manipulation planning, we create a suite of over 2500 cluttered robot manipulation tasks. We find that ARM reconstructions improve the task success rate by 42% over the baseline and are robust to high levels of occlusion.

#### This repository contains code for:
1) Using ARM to create a mujoco simulation of a scene from a depth image for robotics and model-based applications
2) Benchmarking 3D reconstruction algorithms on our robotics manipulation task suite
3) Training and benchmarking 3D reconstruction algorithms on a cluttered scene dataset
4) Replicating and extending the results in our paper

## Using ARM to Reconstruct Scenes
Use 'reconstruct_scene.py' to produce a mujoco .xml of a reconstruction from an rgb image, a depth image, and a mujoco simulation including at a minimum the camera (parameters and pose) used to capture the rgb and depth images.

## 3D Reconstruction Robotics Benchmark
One important application of 3D reconstructions is creating simulations of scenes for robot planning and learning. Tradition medtrics, like chamfer distance, do not nessecarily capture performance on this task: an object may have low chamfer distance but be unstable and tumble away in simulation for example. We present a benchmark of over 2500 robot manipulation tasks to enable benchmarking 3D reconstruction algorithms on this important application. Our benchmark works as follows:
1) Task (robot, objects, rewards) is loaded
2) RGBD image and segmentations are sent to reconstruction algorithm, which send back mesh reconstructions
3) Meshes are postprocessed into simulation with reconstructed objects
4) Robot plans in reconstructed simulation using MPPI and executes in ground truth simulation
5) Task performance in ground truth simulation is recorded

Using your 3D reconstruction algorithm with our benchmark is simple. Just implement the `predict_voxels` method of [`genre/trajopt/sandbox/examples/custom_recon_net.py`](https://github.com/wagnew3/Amodal-3D-Reconstruction-for-Robotic-Manipulationvia-Stability-and-Connectivity/blob/CORL_2020_Code_Release/trajopt/sandbox/examples/custom_recon_net.py) to predict meshes given the per-object four channel voxel input and rgbd image and segmentation masks.

## Cluttered 3D Reconstruction Benchmark
<p align="center"><img src="https://github.com/wagnew3/Amodal-3D-Reconstruction-for-Robotic-Manipulationvia-Stability-and-Connectivity/blob/CORL_2020_Code_Release/webpage_assets/rgb_00008.jpeg" width="250" /><img src="https://github.com/wagnew3/Amodal-3D-Reconstruction-for-Robotic-Manipulationvia-Stability-and-Connectivity/blob/CORL_2020_Code_Release/webpage_assets/rgb_00014.jpeg" width="250" /><img src="https://github.com/wagnew3/Amodal-3D-Reconstruction-for-Robotic-Manipulationvia-Stability-and-Connectivity/blob/CORL_2020_Code_Release/webpage_assets/rgb_00015.jpeg" width="250" /></p>

We provide a high quality synthetic cluttered dataset of over 2.5 million cluttered tabletop reconstruction instances generated from shapenet, and a test set generated from held out shapenet instances. Instances may be accessed either from the raw data, using our dataloader, or using a preprocessed hdf5 file (reccomended).

#### Loading from raw data
We generate our reconstruction instances by taking multiple views of scenes. For each scene, the pose, scale, and model of each object is stored, and for each view the camera information is stored, along with an rgb image, a depth image, and segmentation masks. For more details and examples on loading this format, look at our dataloader [`genre/datasets/shapenet_4_channel.py`](https://github.com/wagnew3/Amodal-3D-Reconstruction-for-Robotic-Manipulationvia-Stability-and-Connectivity/blob/CORL_2020_Code_Release/genre/datasets/shapenet_4_channel.py).

#### Loading with our dataloader
We provide a pytorch dataloader [`genre/datasets/shapenet_4_channel.py`]() that loads and preprocesses reconstruction instances in the method `__getitem__`. Lines 201-220 of [`genre/train.py`](https://github.com/wagnew3/Amodal-3D-Reconstruction-for-Robotic-Manipulationvia-Stability-and-Connectivity/blob/CORL_2020_Code_Release/genre/train.py) show example usage. Modifying this dataloader is the easiest way to modify our four channel representation.

#### HDF5 File
Coming soon!

## Installation
1. Install CUDA 10.0
2. Install mujoco (https://www.roboti.us/download/mujoco200_linux.zip) by extracting into ~/.mujoco/mujoco200_linux (for more guidance see https://github.com/openai/mujoco-py)

#### start ARM env setup
git clone https://github.com/wagnew3/Amodal-3D-Reconstruction-for-Robotic-Manipulationvia-Stability-and-Connectivity.git<br>
cd Amodal-3D-Reconstruction-for-Robotic-Manipulationvia-Stability-and-Connectivity<br>
conda env create --name ARM --file=environment.yml<br>
conda activate ARM<br>

#### install pytorch 0.4
cd ..<br>
git clone https://github.com/pytorch/pytorch.git<br>
cd pytorch<br>
git checkout 0.4_CUDA_10.0_build<br>
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}<br>
export CUDA_ROOT=your cuda install location<br>
export CUDA_HOME=your cuda install location<br>
export CUDA_PATH=your cuda install location<br>
conda install -c anaconda cudnn<br>
FULL_CAFFE2=1 NO_DISTRIBUTED=1 python setup.py install<br>

#### rendering install
conda install -c clinicalgraphics osmesa<br>
export C_INCLUDE_PATH="<your conda install location>/anaconda3/envs/py36/include:$C_LIBRARY_PATH"<br>
  
#### download full shapenet objects
https://www.shapenet.org/

#### download pretrained network weights
Coming soon!

#### [download cluttered reconstruction dataset](https://drive.google.com/file/d/1iNmJvfSX7r_ImQRgH6O1wi6Y8zbUJEUX/view?usp=sharing)

## Replicating Results in Paper

#### Train ARM Network
python train.py --gpu=0,1 --manual_seed=0 --expr_id=0 --suffix=2 --epoch=50 --dataset=shapenet_4_channel --net=genre_given_depth_4_channel --logdir=/raid/wagnew3/ssc_results/logs --dataset_root=&lt;cluttered reconstruction dataset folder&gt; --shapenet_root=&lt;shapenet folder&gt; --workers=40 --batch_size=16 --lr=2e-4 --epoch_batches=2000 --eval_batches=125 --compute_chamfer_dist=0 --stability_loss=1e-8 --connectivity_loss=1e-9

#### Evauluate ARM Network on Reconstruction CD
python train.py --gpu=0,1 --manual_seed=0 --expr_id=0 --resume=13 --suffix=2 --epoch=50 --dataset=shapenet_4_channel --net=genre_given_depth_4_channel --logdir=/raid/wagnew3/ssc_results/logs --dataset_root=&lt;cluttered reconstruction dataset folder&gt; --shapenet_root=&lt;shapenet folder&gt; --workers=40 --batch_size=32 --lr=4e-4 --eval_mode=test --compute_chamfer_dist=1 --eval_at_start --eval_batches=313

#### Evaulate ARM Network on Robot Manipukation Experiments
OMP_NUM_THREADS=1 MUJOCO_GL=osmesa python herb_pushing_mppi.py --top_dir=&lt;cluttered reconstruction dataset folder=&lt;path to github code&gt;/
Amodal-3D-Reconstruction-for-Robotic-Manipulationvia-Stability-and-Connectivity/ --num_cpu=20 --gpu_num=0 --save_dir=&lt;path to results save directory&gt; --vis_while_running=0 --run_num=1 --use_gt=0 --ground_truth=0 --steps=200 --paths_per_cpu=25  --recon_net=&lt;path to saved network weights&gt; --make_video=1 --remove_old_runs=1
