from matplotlib import pyplot as plt
import csv

with open('./results/2d_eval.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    header1 = next(rows)
    header2 = next(rows)
    inp_psnr = []
    inp_ssim = []
    inp_rmse = []
    pre_psnr_2d = []
    pre_ssim_2d = []
    pre_rmse_2d = []
    for row in rows:
        inp_psnr.append(round(float(row[1]), 2))
        inp_ssim.append(round(float(row[3]), 3))
        inp_rmse.append(round(float(row[5]), 5))
        pre_psnr_2d.append(round(float(row[7]), 2))
        pre_ssim_2d.append(round(float(row[9]), 3))
        pre_rmse_2d.append(round(float(row[11]), 5))

with open('./results/3d_eval.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    header1 = next(rows)
    header2 = next(rows)
    inp_psnr_3d = []
    inp_ssim_3d = []
    inp_rmse_3d = []
    pre_psnr_3d = []
    pre_ssim_3d = []
    pre_rmse_3d = []
    for row in rows:
        pre_psnr_3d.append(round(float(row[7]), 2))
        pre_ssim_3d.append(round(float(row[9]), 3))
        pre_rmse_3d.append(round(float(row[11]), 5))

plt.plot([i for i in range(1,101)], inp_psnr, '-^', markersize=3)
plt.plot([i for i in range(1,101)], pre_psnr_2d, '-o', markersize=3)
plt.plot([i for i in range(1,101)], pre_psnr_3d)
plt.ylabel('PSNR(dB)')
plt.xlabel('epoch')
plt.legend(['STF23', '2D U-net', '3D U-net'])
plt.savefig("./results/PSNR")
plt.close()

plt.plot([i for i in range(1,101)], inp_ssim, '-^', markersize=3)
plt.plot([i for i in range(1,101)], pre_ssim_2d, '-o', markersize=3)
plt.plot([i for i in range(1,101)], pre_ssim_3d)
plt.ylabel('SSIM')
plt.xlabel('epoch')
plt.legend(['STF23', '2D U-net', '3D U-net'])
plt.savefig("./results/SSIM")
plt.close()

plt.plot([i for i in range(1,101)], inp_rmse, '-^', markersize=3)
plt.plot([i for i in range(1,101)], pre_rmse_2d, '-o', markersize=3)
plt.plot([i for i in range(1,101)], pre_rmse_3d)
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.legend(['STF23', '2D U-net', '3D U-net'])
plt.savefig("./results/RMSE")
plt.close()

