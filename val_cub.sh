
####CUB

# 224-trt
#python ./tools_cam/test_cam.py --config_file configs/CUB/deit_trt_base_patch16_224_0.6.yaml --resume ./ckpt_save/CUB/deit_trt_base_patch16_224_TOKENTHR0.6_BS128_0.9196/ckpt/model_best_top1_loc.pth MODEL.CAM_THR 0.1

# 384-trt
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python ./tools_cam/test_cam.py --config_file configs/CUB/deit_trt_base_patch16_384_0.65.yaml --resume ./ckpt_save/CUB/deit_trt_base_patch16_384_TOKENTHR0.65_BS128_0.946/ckpt/model_best_top1_loc.pth MODEL.CAM_THR 0.1 BASIC.DISP_FREQ 10 

# 224-trt_fuse
#export CUDA_VISIBLE_DEVICES=0,1,2,3
# python ./tools_cam/test_cam.py --config_file configs/CUB/deit_trt_fuse_base_patch16_224_0.6.yaml --resume ./ckpt_save/CUB/deit_trt_fuse_base_patch16_224_TOKENTHR0.6_BS128_0.912/ckpt/model_best_top1_loc.pth MODEL.CAM_THR 0.1 TEST.METRICS gt_top  #(TEST.METRICS maxboxaccv2)

# 384-trt_fuse
#python ./tools_cam/test_cam.py --config_file configs/CUB/deit_trt_fuse_base_patch16_384_0.65.yaml --resume ./ckpt_save/CUB/deit_trt_fuse_base_patch16_384_TOKENTHR0.65_BS128_0.941/ckpt/model_best_top1_loc.pth MODEL.CAM_THR 0.1 TEST.METRICS gt_top BASIC.DISP_FREQ 10 #TEST.SAVE_BOXED_IMAGE True



#### example of other configs ####

## deit
# python ./tools_cam/test_cam.py --config_file ./configs/CUB/deit_base_patch16_224.yaml --resume ./model.pth MODEL.CAM_THR 0.1 

## ts-cam 
#python ./tools_cam/test_cam.py --config_file configs/CUB/deit_tscam_base_patch16_224.yaml --resume ./TSCAM_resumes/ts-cam-deit-base/CUB-200-2011/model_epoch60.pth MODEL.CAM_THR 0.1 TEST.SAVE_BOXED_IMAGE True

## deit_pos
#python ./tools_cam/test_cam.py --config_file configs/CUB/deitpos_tscam_base_patch16_224_s65.yaml --resume model.pth MODEL.CAM_THR 0.1 #TEST.SAVE_BOXED_IMAGE True

# (green boxes mean test, red boxes mean ground truth)