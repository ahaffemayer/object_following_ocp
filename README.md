# Object Trajectory Following with OCP

## Generating the grasping poses

### Cloning the repos
```bash
git clone https://github.com/NVlabs/GraspGen.git 
git clone https://huggingface.co/adithyamurali/GraspGenModels # Those are the checkpoints
```

### Building the docker
```bash
cd GraspGen && bash docker/build.sh
pip install -e .                  
```

> [!NOTE]
> If you have this `fatal: detected dubious ownership` error while doing the pip install, use the next command instead.

```bash
git config --global --add safe.directory /code/GraspGen && 
pip install -e .                  
```

### Running the script to generate the grasping poses
To verify that the docker properly works, you can run this script:
```bash
python /code/GraspGen/scripts/demo_object_mesh.py --mesh_file /code/GraspGenModels/sample_data/meshes/box.obj --mesh_scale 0.1 --gripper_config /code/GraspGenModels/checkpoints/graspgen_franka_panda.yml --output_file /code/GraspGenModels/test.yml
```
In another terminal, create a `meshcat-server` and run it.

The grasping poses are then in the `test.yml` file.
You simply need to change the .obj, scale and where you want to store the results. 

## Generating the trajectories

To use this repo, a devcontainer was created. Simply launch code in the root folder and chose "Open with devcontainer".

### Processing the poses
#### Filtering the poses
Now that you have the grasping poses (if you didn't change the parameters, 100 poses are generated), we need to extract 1 - 5 distinct poses. To do so, we use the Further Point Sampling algorithm to select interesting ones:

```bash
python scripts/parser_grasps.py --input_yaml pathtofile --output_yaml pathtofile --num_grasps ngrasps
```
#### Visualizing the poses
If you want to visualize the poses, launch: 
```bash
python scripts/show_grasps.py --input_yaml pathtofile --mesh_path pathtomesh
```
> [!NOTE]
> If the script is launched but doesn't respond, launch a `meshcat-server` in another terminal in the devcontainer.


### Generating the trajectories

Launch this script to run the grid search: 

```bash
python example/robot_motion/main_grid_search.py   --object-traj path_to_json  --scale-path path_to_scale   --config-path /workspaces/object_following_ocp/example/robot_motion/configs/ocp_config_panda.yml   --top-n 3
```
`--scale-path` is not necessary, it's only if you want to tweak the scales. 


