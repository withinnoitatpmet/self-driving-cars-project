#! /usr/bin/python3
from glob import glob
import numpy as np
import math
from scipy import misc


files = glob('deploy/*/*/*_image.jpg')
# idx = np.random.randint(0, len(files))
for idx in range(len(files)):
    print("Pre-processing: "+str(idx+1)+" of "+str(len(files))+" in total...")
    snapshot = files[idx]
    # print(snapshot)
    lidar = np.zeros([1052,1914],dtype=np.float32)

    xyz = np.fromfile(snapshot.replace('_image.jpg', '_cloud.bin'), dtype=np.float32)
    xyz.resize([3, xyz.size // 3])

    proj = np.fromfile(snapshot.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
    proj.resize([3, 4])

    uv = np.dot(proj, np.vstack([xyz, np.ones_like(xyz[0, :])]))
    uv = uv / uv[2, :]

    dist = np.linalg.norm(xyz, axis=0)

    for i in range(len(dist)):
        x,y,z = xyz[:,i]
        r = dist[i]
        dx = r*math.tan(0.1/180.0*math.pi)
        dy = r*math.tan(0.2/180.0*math.pi)
        pmax=np.dot(proj,np.array([x+dx,y+dy,z,1]))
        pmin=np.dot(proj,np.array([x-dx,y-dy,z,1]))
        xmin,ymin,zmin = pmin/pmin[2]
        xmax,ymax,zmax = pmax/pmax[2]
        for u in range(int(round(xmin)-1),int(round(xmax))+1):
            for v in range(int(round(ymin)-1),min(1052,int(round(ymax)+1))):
                lidar[v-1,u-1] = r


    max_finite_distance = np.max(lidar)
    lidar = np.where(lidar>0,max_finite_distance+1-lidar,0)


    # fig1 = plt.figure(1, figsize=(16, 9))
    # ax1 = fig1.add_subplot(1, 1, 1)
    # ax1.imshow(lidar)
    # # ax1.scatter(uv[0, :], uv[1, :], c=dist, marker='.', s=1)
    # ax1.axis('scaled')
    # fig1.tight_layout()

    # plt.show()

    # write to file
    # misc.imsave("test.png",lidar/np.max(lidar)*255)
    misc.imsave(snapshot.replace('_image.jpg', '_lidar.png'), lidar/np.max(lidar)*255)

