import numpy as np
import open3d as o3d
import cv2

def process_gemini_depth_to_birdview(depth_image, intrinsic_matrix, camera_pose_extrinsic):
    """
    Gemini 335Lの深度データから点群を生成し、車両座標系に変換して正射投影図（バードアイビュー）を生成する。
    
    Parameters:
    - depth_image: 16-bit unsigned の深度画像 (単位: mm)
    - intrinsic_matrix: カメラの内部パラメータ (3x3 numpy array)
    - camera_pose_extrinsic: 車両基準から見たカメラの外部パラメータ (4x4 外部行列 [R | t])
    """
    height, width = depth_image.shape
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    # 1. Open3DのImageオブジェクトに変換して点群(PointCloud)を生成
    depth_o3d = o3d.geometry.Image(depth_image)
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    # 深度をメートル単位にスケーリング (Orbbecデフォルトは通常1mm = 1000)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d, intrinsic_o3d, depth_scale=1000.0, depth_trunc=15.0
    )

    # 2. 車両座標系（あるいは運転席FPCP基準）への座標変換
    # camera_pose_extrinsic を用いてカメラ座標系から車両座標系へ変換
    pcd.transform(camera_pose_extrinsic)
    
    points = np.asarray(pcd.points) # 形: (N, 3) [X: 左右, Y: 上下, Z: 前後] が一般的
    
    if len(points) == 0:
        print("有効な点群が取得できませんでした。")
        return None

    # 3. バードアイビュー（正射投影）の描画パラメータ設定
    # 前方 0m ~ 15m, 左右 -6m ~ +6m の範囲を俯瞰表示する場合
    x_min, x_max = -6.0, 6.0
    z_min, z_max = 0.0, 15.0
    resolution = 0.05  1ピクセル = 5cm
    
    img_width = int((x_max - x_min) / resolution)
    img_height = int((z_max - z_min) / resolution)
    
    bird_view = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    # 4. 点群を2Dグリッド（トップビュー）にマッピング
    # 車両座標系: X(左右), Z(前後) を利用
    x_coords = points[:, 0]
    z_coords = points[:, 2]
    y_coords = points[:, 1]  # 高さ (障害物の判定に使用可能)

    # 範囲内の点だけをフィルタリング
    valid_idx = (x_coords >= x_min) & (x_coords <= x_max) & (z_coords >= z_min) & (z_coords <= z_max)
    
    x_filtered = x_coords[valid_idx]
    z_filtered = z_coords[valid_idx]
    y_filtered = y_coords[valid_idx]

    # ピクセル座標への変換
    u = ((x_filtered - x_min) / resolution).astype(int)
    # Z軸（前方）は画像の上方向に向かうため反転
    v = (img_height - 1 - ((z_filtered - z_min) / resolution)).astype(int)

    # 範囲内に収まるインデックスを再度絞り込み
    pixel_valid = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
    u = u[pixel_valid]
    v = v[pixel_valid]
    y_filtered = y_filtered[pixel_valid]

    # 5. 高さに応じたカラーリング（地面付近は青・高い位置は赤など）でプロット
    # JIS A 8311評価用：高さ1.5mの試験体などがどこにあるかを可視化
    for px, pz, py in zip(u, v, y_filtered):
        # 高さに応じた色づけ (例: 地面に近いところは暗く、高さがあるものは明るく)
        color_val = int(np.clip((py + 1.0) * 100, 50, 255))
        bird_view[pz, px] = [color_val, 255 - color_val, 150]

    # 6. JIS A 8311 規定の距離グリッド（例: 半径1mの機側境界や数mごとの同心円/ライン）を描画
    # 車両中心 (X=0) の位置を算出
    center_u = int((0.0 - x_min) / resolution)
    
    # 車両の真下（Z=0）付近に基準線を引く
    cv2.line(bird_view, (center_u, img_height), (center_u, 0), (200, 200, 200), 1)
    
    # 1mごとにグリッドとテキストを描画
    for dist in range(1, int(z_max)):
        v_pos = int(img_height - 1 - (dist / resolution))
        cv2.putText(bird_view, f"{dist}m", (10, v_pos - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.line(bird_view, (0, v_pos), (img_width, v_pos), (50, 50, 50), 1)

    return bird_view

# --- 実行用のダミーデータテスト例 ---
if __name__ == "__main__":
    # 仮想的なインサイドパラメータ (例: 640x480解像度)
    intrinsic = np.array([
        [600.0, 0.0, 320.0],
        [0.0, 600.0, 240.0],
        [0.0,   0.0,   1.0]
    ])
    
    # カメラ外部パラメータ（車両前方に取り付け、高さ2m、下向きに若干傾いている仮定の4x4行列）
    extrinsic = np.eye(4)
    # 例: Y軸方向に +2.0m（屋根の高さ）、Z軸方向に前方に配置する場合の回転・平行移動
    extrinsic[1, 3] = 2.0  # 高さ 2m
    
    # ダミーの深度画像作成 (640x480, 単位mm)
    dummy_depth = np.ones((480, 640), dtype=np.uint16) * 3000  # 全面3m先にある壁を仮定
    # 中央付近に障害物を置く
    dummy_depth[200:300, 250:350] = 1500  # 1.5m先の手前障害物
    
    # 実行
    bird_eye_image = process_gemini_depth_to_birdview(dummy_depth, intrinsic, extrinsic)
    
    if bird_eye_image is not None:
        cv2.imshow("JIS A 8311 Blind Spot Bird-Eye View", bird_eye_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
