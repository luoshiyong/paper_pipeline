import os
img_path = "/home/luosy/data3382/image/"
mask_path =  "/home/luosy/data3382/mask/"
val_img_path =  "/home/luosy/data3382/val_image/"
val_mask_path =  "/home/luosy/data3382/val_mask/"
index = [ 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 101, 105, 109, 113, 117, 121, 125, 129]
val_index = [1,4,8,14,22,25,41,64,67,74,85,95,97,118,130]  # 15
# 75
for idx in val_index:
    cmd_dd1 = "mv /home/luosy/data3382/image/{}_* /home/luosy/data3382/val_image/".format(idx)
    cmd_dd2 = "mv /home/luosy/data3382/mask/{}_* /home/luosy/data3382/val_mask/".format(idx)
    os.system(cmd_dd1)
    os.system(cmd_dd2)
