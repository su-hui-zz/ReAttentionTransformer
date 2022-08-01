export CUDA_VISIBLE_DEVICES=6

# 224-trt
#python ./tools_cam/test_cam.py --config_file configs/ILSVRC/deit_trt_base_patch16_224_0.95.yaml --resume ./ckpt_save/ImageNet/deit_trt_base_patch16_224_0.95_0.688/ckpt/model_best_top1_loc.pth MODEL.CAM_THR 0.12

# 224-trt-fuse
#export CUDA_VISIBLE_DEVICES=0,1,2,3
# python ./tools_cam/test_cam.py --config_file configs/ILSVRC/deit_trt_fuse_base_patch16_224_0.95.yaml --resume ./ckpt_save/ImageNet/deit_trt_fuse_base_patch16_224_0.707/ckpt/model_best_top1_loc.pth MODEL.CAM_THR 0.12 TEST.METRICS gt_top  #(TEST.METRICS maxboxaccv2)


#### example of other configs ####
# ts-cam
# python ./tools_cam/test_cam.py --config_file ./configs/ILSVRC/deit_tscam_small_patch16_224.yaml --resume ./TSCAM_resumes/ts-cam-deit-small/ILSVRC2012-20220412T021547Z-001/ILSVRC2012/model_epoch12.pth  MODEL.CAM_THR 0.12 # TEST.SAVE_BOXED_IMAGE True 

