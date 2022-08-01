export CUDA_VISIBLE_DEVICES=0,1,2,3

#python ./tools_cam/train_cam.py --config_file ./configs/CUB/deit_trt_base_patch16_224_0.6.yaml --lr 5e-5 MODEL.CAM_THR 0.1
#python ./tools_cam/train_cam_fusecamfz.py --config_file ./configs/CUB/deit_trt_fuse_base_patch16_224_0.6.yaml --lr 5e-5 MODEL.CAM_THR 0.1 MODEL.POSWEIGHTS ./ckpt/CUB/deit_trt_base_patch16_224_TOKENTHR0.6_BS128/ckpt/model_best_top1_loc.pth

#python ./tools_cam/train_cam.py --config_file ./configs/CUB/deit_trt_base_patch16_384_0.65.yaml --lr 5e-5 MODEL.CAM_THR 0.1
#python ./tools_cam/train_cam_fusecamfz.py --config_file ./configs/CUB/deit_trt_fuse_base_patch16_384_0.65.yaml --lr 5e-5 MODEL.CAM_THR 0.1 MODEL.POSWEIGHTS ./ckpt/CUB/deit_trt_base_patch16_384_TOKENTHR0.65_BS128//ckpt/model_best_top1_loc.pth




#### example of other configs ####
#python ./tools_cam/train_cam.py --config_file ./configs/CUB/deit_tscam_base_patch16_224.yaml --lr 5e-5 MODEL.CAM_THR 0.1

#python ./tools_cam/train_cam.py --config_file ./configs/CUB/deitpos_tscam_base_patch16_224_s65.yaml --lr 5e-5 MODEL.CAM_THR 0.1

#python ./tools_cam/train_cam.py --config_file ./configs/CUB/deit_base_patch16_224.yaml --lr 5e-5 MODEL.CAM_THR 0.1

#python ./tools_cam/train_cam.py --config_file ./configs/CUB/vggcam_patch16_224.yaml --lr 5e-5 MODEL.CAM_THR 0.1
