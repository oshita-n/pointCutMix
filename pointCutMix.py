import open3d
import numpy as np

def random_sampling(points, num_points):
    if len(points) >= num_points:
        sampled_points = points[np.random.choice(len(points), num_points, replace=False)]
    else:
        sampled_points = points[np.random.choice(len(points), num_points, replace=True)]
    return sampled_points

if __name__ == "__main__":
    pcd_car = open3d.io.read_point_cloud("car_0003.pcd") #点群ファイルを変数pointcloudに保存
    pcd_airplane = open3d.io.read_point_cloud("airplane_0002.pcd") #点群ファイルを変数pointcloudに保存

    pcd_car = np.array(pcd_car.points)
    pcd_airplane = np.array(pcd_airplane.points)


    pcd_car = random_sampling(pcd_car, 30000)
    pcd_airplane =  random_sampling(pcd_airplane, 30000)

    # x, y, zの最大値を取得
    max_x = np.max(pcd_airplane[:, 0])
    max_y = np.max(pcd_airplane[:, 1])
    max_z = np.max(pcd_airplane[:, 2])
    min_x = np.min(pcd_airplane[:, 0])
    min_y = np.min(pcd_airplane[:, 1])
    min_z = np.min(pcd_airplane[:, 2])

    car_max_x = np.max(pcd_car[:, 0])
    car_max_y = np.max(pcd_car[:, 1])
    car_max_z = np.max(pcd_car[:, 2])

    # 乱数で切り取る範囲を決定
    # ベータ分布から乱数を生成
    lam = np.random.beta(1.0, 1.0, 1)[0]
    print("lam: ", lam)
    bx = np.random.randint(min_x, max_x, 1)
    by = np.random.randint(min_y, max_y, 1)
    bz = np.random.randint(min_z, max_z, 1)

    print("bx: ", bx)
    print("by: ", by)
    print("bz: ", bz)
    H = np.random.randint(0, max_y, 1)
    H = H * np.sqrt(1-lam)
    W = np.random.randint(0, max_x, 1)
    W = W * np.sqrt(1-lam)
    DEPTH = np.random.randint(0, max_z, 1)
    print("H: ", H)
    print("W: ", W)
    print("DEPTH: ", DEPTH)

    center = np.array([bx[0], by[0], bz[0]])
    size = np.array([H[0], W[0], DEPTH[0]]) 
    B = np.where(np.all(pcd_airplane > center - size, axis=1) & np.all(pcd_airplane < center + size, axis=1), 0, 1)
    B = np.tile(B, (3, 1, 1)).reshape(30000, 3)
    # 単位行列の作成
    In = np.ones((30000, 3))
    class_x1 = np.array([0, 1])
    class_x2 = np.array([1, 0])
    x_new =  B* pcd_car + (In - B) * pcd_airplane
    y_new = lam * class_x1 + (1 - lam) * class_x2
    # numpy array からpcdに変換
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(x_new)
    pcd.colors = open3d.utility.Vector3dVector(np.array([[0, 0, 128] for i in range(len(x_new))]))
    print(y_new)
    open3d.visualization.draw_geometries([pcd])