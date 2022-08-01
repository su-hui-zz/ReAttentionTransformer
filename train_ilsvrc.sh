
export CUDA_VISIBLE_DEVICES=0,1,2,3
#export CUDA_VISIBLE_DEVICES=4,5,6,7
#python ./tools_cam/train_cam.py --config_file ./configs/ILSVRC/deit_tscam_base_patch16_224.yaml --lr 5e-4 MODEL.CAM_THR 0.12

#python ./tools_cam/train_cam.py --config_file ./configs/ILSVRC/deitpos_tscam_base_patch16_224_s150.yaml --lr 5e-4 MODEL.CAM_THR 0.1


#python ./tools_cam/train_cam.py --config_file ./configs/ILSVRC/deit_trt_base_patch16_224_0.95.yaml --lr 5e-4 MODEL.CAM_THR 0.12

#python ./tools_cam/train_cam_fusecamfz.py --config_file ./configs/ILSVRC/deit_trt_fuse_base_patch16_224_0.95.yaml --lr 5e-4 MODEL.CAM_THR 0.12




