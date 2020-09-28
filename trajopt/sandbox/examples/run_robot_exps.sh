#!/bin/bash
#GT
#PATH=/home/wagnew3/github/ssc${PATH:+:${PATH}}$ CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MUJOCO_GL=osmesa PYTHONPATH=/home/wagnew3/github/ssc python herb_pushing_mppi.py --top_dir=/home/wagnew3/github/ssc/ --num_cpu=80 --gpu_num=0 --save_dir=/raid/wagnew3/ssc_results/ground_truth_mppi/ --vis_while_running=0 --run_num=0 --total_runs=1 --ground_truth=1 --steps=150 --recon_net=/raid/wagnew3/ssc_results/logs/genre_given_depth_shapenet_table_norm_2/33/nets/0005.pt --recon_net_type=genre_given_depth --paths_per_cpu=25 --use_args_cpus=1 --make_video=0 --seed=0

#Baseline
PATH=/home/wagnew3/github/ssc${PATH:+:${PATH}}$ CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 MUJOCO_GL=osmesa PYTHONPATH=/home/wagnew3/github/ssc python herb_pushing_mppi.py --top_dir=/home/wagnew3/github/ssc/ --num_cpu=80 --gpu_num=7 --save_dir=/raid/wagnew3/ssc_results/baseline_mppi/ --vis_while_running=0 --run_num=0 --total_runs=1 --ground_truth=0 --steps=150 --recon_net=/raid/wagnew3/ssc_results/logs/genre_given_depth_shapenet_table_norm_2/33/nets/0005.pt --recon_net_type=genre_given_depth --paths_per_cpu=25 --use_args_cpus=1 --make_video=0 --seed=0

#Baseline+Extrusion
PATH=/home/wagnew3/github/ssc${PATH:+:${PATH}}$ CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 MUJOCO_GL=osmesa PYTHONPATH=/home/wagnew3/github/ssc python herb_pushing_mppi.py --top_dir=/home/wagnew3/github/ssc/ --num_cpu=80 --gpu_num=7 --save_dir=/raid/wagnew3/ssc_results/extrusion_baseline_mppi/ --vis_while_running=0 --run_num=0 --total_runs=1 --ground_truth=0 --steps=150 --recon_net=/raid/wagnew3/ssc_results/logs/genre_given_depth_shapenet_table_norm_2/33/nets/0005.pt --recon_net_type=genre_given_depth --paths_per_cpu=25 --use_args_cpus=1 --extrusion_baseline=1 --make_video=0 --seed=0

#Four Channel+Stability
PATH=/home/wagnew3/github/ssc${PATH:+:${PATH}}$ CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 MUJOCO_GL=osmesa PYTHONPATH=/home/wagnew3/github/ssc python herb_pushing_mppi.py --top_dir=/home/wagnew3/github/ssc/ --num_cpu=80 --gpu_num=7 --save_dir=/raid/wagnew3/ssc_results/stability_four_channel_mppi/ --vis_while_running=0 --run_num=0 --total_runs=1 --ground_truth=0 --steps=150 --recon_net=/raid/wagnew3/ssc_results/logs/genre_given_depth_shapenet_table_norm_2/41/nets/0007.pt --recon_net_type=genre_given_depth_4_channel --paths_per_cpu=25 --use_args_cpus=1 --make_video=0 --seed=0

#Four Channel
PATH=/home/wagnew3/github/ssc${PATH:+:${PATH}}$ CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 MUJOCO_GL=osmesa PYTHONPATH=/home/wagnew3/github/ssc python herb_pushing_mppi.py --top_dir=/home/wagnew3/github/ssc/ --num_cpu=80 --gpu_num=7 --save_dir=/raid/wagnew3/ssc_results/four_channel_mppi/ --vis_while_running=0 --run_num=0 --total_runs=1 --ground_truth=0 --steps=150 --recon_net=/raid/wagnew3/ssc_results/logs/genre_given_depth_shapenet_table_norm_2/34/nets/0003.pt --recon_net_type=genre_given_depth_4_channel --paths_per_cpu=25 --use_args_cpus=1 --make_video=0 --seed=0

#GT
#PATH=/home/wagnew3/github/ssc${PATH:+:${PATH}}$ CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MUJOCO_GL=osmesa PYTHONPATH=/home/wagnew3/github/ssc python herb_pushing_mppi.py --top_dir=/home/wagnew3/github/ssc/ --num_cpu=80 --gpu_num=0 --save_dir=/raid/wagnew3/ssc_results/ground_truth_mppi/ --vis_while_running=0 --run_num=0 --total_runs=1 --ground_truth=1 --steps=150 --recon_net=/raid/wagnew3/ssc_results/logs/genre_given_depth_shapenet_table_norm_2/33/nets/0005.pt --recon_net_type=genre_given_depth --paths_per_cpu=25 --use_args_cpus=1 --make_video=0 --seed=1

#Baseline
PATH=/home/wagnew3/github/ssc${PATH:+:${PATH}}$ CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 MUJOCO_GL=osmesa PYTHONPATH=/home/wagnew3/github/ssc python herb_pushing_mppi.py --top_dir=/home/wagnew3/github/ssc/ --num_cpu=80 --gpu_num=7 --save_dir=/raid/wagnew3/ssc_results/baseline_mppi/ --vis_while_running=0 --run_num=0 --total_runs=1 --ground_truth=0 --steps=150 --recon_net=/raid/wagnew3/ssc_results/logs/genre_given_depth_shapenet_table_norm_2/33/nets/0005.pt --recon_net_type=genre_given_depth --paths_per_cpu=25 --use_args_cpus=1 --make_video=0 --seed=1

#Baseline+Extrusion
PATH=/home/wagnew3/github/ssc${PATH:+:${PATH}}$ CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 MUJOCO_GL=osmesa PYTHONPATH=/home/wagnew3/github/ssc python herb_pushing_mppi.py --top_dir=/home/wagnew3/github/ssc/ --num_cpu=80 --gpu_num=3 --save_dir=/raid/wagnew3/ssc_results/extrusion_baseline_mppi/ --vis_while_running=0 --run_num=0 --total_runs=1 --ground_truth=0 --steps=150 --recon_net=/raid/wagnew3/ssc_results/logs/genre_given_depth_shapenet_table_norm_2/33/nets/0005.pt --recon_net_type=genre_given_depth --paths_per_cpu=25 --use_args_cpus=1 --extrusion_baseline=1 --make_video=0 --seed=1

#Four Channel+Stability
PATH=/home/wagnew3/github/ssc${PATH:+:${PATH}}$ CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 MUJOCO_GL=osmesa PYTHONPATH=/home/wagnew3/github/ssc python herb_pushing_mppi.py --top_dir=/home/wagnew3/github/ssc/ --num_cpu=80 --gpu_num=7 --save_dir=/raid/wagnew3/ssc_results/stability_four_channel_mppi/ --vis_while_running=0 --run_num=0 --total_runs=1 --ground_truth=0 --steps=150 --recon_net=/raid/wagnew3/ssc_results/logs/genre_given_depth_shapenet_table_norm_2/41/nets/0007.pt --recon_net_type=genre_given_depth_4_channel --paths_per_cpu=25 --use_args_cpus=1 --make_video=0 --seed=1

#Four Channel
PATH=/home/wagnew3/github/ssc${PATH:+:${PATH}}$ CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 MUJOCO_GL=osmesa PYTHONPATH=/home/wagnew3/github/ssc python herb_pushing_mppi.py --top_dir=/home/wagnew3/github/ssc/ --num_cpu=80 --gpu_num=7 --save_dir=/raid/wagnew3/ssc_results/four_channel_mppi/ --vis_while_running=0 --run_num=0 --total_runs=1 --ground_truth=0 --steps=150 --recon_net=/raid/wagnew3/ssc_results/logs/genre_given_depth_shapenet_table_norm_2/34/nets/0003.pt --recon_net_type=genre_given_depth_4_channel --paths_per_cpu=25 --use_args_cpus=1 --make_video=0 --seed=1




