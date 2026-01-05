import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import math
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
import warnings
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
plt.style.use('ggplot')

class InteractiveCurveFitter:
    """交互式曲线拟合类（使用matplotlib）"""
    
    def __init__(self, image):
        self.image = image
        self.img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        self.height, self.width = self.img_rgb.shape[:2]
        
        self.fitted_curves = []  # 存储所有拟合的曲线
        self.user_points = []  # 存储用户点击的点
        self.current_curve = None  # 当前拟合的曲线
        
        # 创建结果图像
        self.result_with_bg = self.img_rgb.copy()
        self.curves_only = np.zeros_like(self.img_rgb)  # 黑色背景
        
        # matplotlib图形和坐标轴
        self.fig = None
        self.ax = None
        
    def onclick(self, event):
        """
        鼠标点击事件处理
        """
        if event.inaxes == self.ax:
            x, y = int(event.xdata), int(event.ydata)
            self.user_points.append((x, y))
            print(f"已选择点 ({x}, {y})，当前曲线有 {len(self.user_points)} 个点")
            
            # 更新显示
            self.update_display()
            
            # 当有至少3个点时，拟合曲线
            if len(self.user_points) >= 3:
                self.fit_curve_to_points()
                self.update_display()
    
    def onkey(self, event):
        """
        键盘事件处理
        """
        if event.key == 'enter':
            # 保存当前拟合的曲线
            if self.current_curve is not None:
                self.fitted_curves.append(self.current_curve)
                curve_points = self.current_curve
                print(f"已保存曲线 {len(self.fitted_curves)}: 包含 {len(curve_points)} 个点")
                
                # 在结果图像上绘制最终曲线（白色），线宽增加2个像素
                self.draw_curve_on_image(self.result_with_bg, curve_points, color=(255, 255, 255), thickness=4)  # 原厚度2+2=4
                self.draw_curve_on_image(self.curves_only, curve_points, color=(255, 255, 255), thickness=4)    # 原厚度2+2=4
                
                # 清空当前点集，准备下一个曲线
                self.user_points = []
                self.current_curve = None
                
                self.update_display()
            else:
                print("没有可保存的曲线")
                
        elif event.key == 'c':
            # 清除当前点集
            self.user_points = []
            self.current_curve = None
            print("已清除当前点集")
            self.update_display()
            
        elif event.key == 'd':
            # 删除最后一个点
            if self.user_points:
                removed_point = self.user_points.pop()
                print(f"已删除点: {removed_point}")
                
                # 如果有足够点数，重新拟合曲线
                if len(self.user_points) >= 3:
                    self.fit_curve_to_points()
                else:
                    self.current_curve = None
                    
                self.update_display()
            else:
                print("没有可删除的点")
                
        elif event.key == 'f':
            # 完成所有曲线的拟合
            if self.fitted_curves:
                print(f"已完成 {len(self.fitted_curves)} 个曲线的拟合")
                plt.close()
            else:
                print("请先拟合至少一个曲线")
                
        elif event.key == 'q':
            # 退出
            plt.close()
    
    def fit_curve_to_points(self):
        """
        使用三次样条插值拟合光滑曲线到点集
        """
        if len(self.user_points) < 3:
            return
        
        # 将点转换为numpy数组
        points = np.array(self.user_points)
        x = points[:, 0]
        y = points[:, 1]
        
        try:
            # 计算累积弦长作为参数
            chord_lengths = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
            t = np.concatenate(([0], np.cumsum(chord_lengths)))
            
            # 使用三次样条插值
            cs_x = CubicSpline(t, x)
            cs_y = CubicSpline(t, y)
            
            # 生成更密集的参数值以获得光滑曲线
            t_dense = np.linspace(t[0], t[-1], 200)
            
            # 计算插值点
            x_dense = cs_x(t_dense)
            y_dense = cs_y(t_dense)
            
            # 确保曲线从第一个点开始，到最后一个点结束
            curve_points = []
            for i in range(len(x_dense)):
                px, py = x_dense[i], y_dense[i]
                # 确保点在图像范围内
                if 0 <= px < self.width and 0 <= py < self.height:
                    curve_points.append((px, py))
            
            # 强制添加第一个点和最后一个点，确保精确性
            if len(curve_points) > 0:
                # 确保第一个点就是用户标注的第一个点
                if len(curve_points) > 1:
                    curve_points[0] = (x[0], y[0])
                # 确保最后一个点就是用户标注的最后一个点
                if len(curve_points) > 1:
                    curve_points[-1] = (x[-1], y[-1])
            
            self.current_curve = curve_points
            print(f"拟合光滑曲线: 包含 {len(curve_points)} 个点")
            
        except Exception as e:
            print(f"曲线拟合失败: {e}")
            # 如果样条插值失败，使用简单的多项式拟合
            self.fit_polynomial_curve(points)
    
    def fit_polynomial_curve(self, points):
        """
        使用多项式拟合作为备选方案
        """
        x = points[:, 0]
        y = points[:, 1]
        
        try:
            # 使用3次多项式拟合
            coeffs = np.polyfit(x, y, min(3, len(x)-1))
            poly = np.poly1d(coeffs)
            
            # 生成x值范围（从第一个点到最后一个点）
            x_min, x_max = min(x), max(x)
            x_dense = np.linspace(x_min, x_max, 200)
            y_dense = poly(x_dense)
            
            curve_points = []
            for i in range(len(x_dense)):
                px, py = x_dense[i], y_dense[i]
                if 0 <= px < self.width and 0 <= py < self.height:
                    curve_points.append((px, py))
            
            # 确保第一个点和最后一个点精确匹配
            if len(curve_points) > 0:
                curve_points[0] = (x[0], y[0])
                curve_points[-1] = (x[-1], y[-1])
            
            self.current_curve = curve_points
            print(f"使用多项式拟合曲线: 包含 {len(curve_points)} 个点")
            
        except Exception as e:
            print(f"多项式拟合也失败: {e}")
            self.current_curve = None
    
    def draw_curve_on_image(self, image, curve_points, color=(255, 255, 255), thickness=4):
        """
        在图像上绘制曲线（白色），线宽增加2个像素
        """
        if len(curve_points) < 2:
            return
        
        # 绘制曲线，线宽增加2个像素（原厚度2+2=4）
        for i in range(len(curve_points) - 1):
            x1, y1 = curve_points[i]
            x2, y2 = curve_points[i+1]
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    
    def update_display(self):
        """
        更新显示图像
        """
        # 清除当前图像
        self.ax.clear()
        
        # 显示原始图像
        self.ax.imshow(self.img_rgb)
        
        # 绘制所有已保存的曲线（白色），线宽增加
        for i, curve in enumerate(self.fitted_curves):
            if curve:
                x_coords = [p[0] for p in curve]
                y_coords = [p[1] for p in curve]
                self.ax.plot(x_coords, y_coords, 'w-', linewidth=3, label=f'曲线 {i+1}')  # 线宽增加
        
        # 绘制当前点集和拟合的曲线
        if self.user_points:
            # 绘制点（绿色）
            for i, point in enumerate(self.user_points):
                self.ax.plot(point[0], point[1], 'go', markersize=8, markeredgecolor='white', markeredgewidth=2)
                self.ax.text(point[0] + 10, point[1] + 10, str(i+1), 
                            color='white', fontsize=12,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.7))
            
            # 绘制点之间的连接线（按照点击顺序，黄色虚线）
            if len(self.user_points) >= 2:
                for i in range(len(self.user_points) - 1):
                    x1, y1 = self.user_points[i]
                    x2, y2 = self.user_points[i+1]
                    self.ax.plot([x1, x2], [y1, y2], 'y--', linewidth=1, alpha=0.7)
            
            # 绘制拟合的曲线（三个点以后每添加一个点都显示，绿色），线宽增加
            if self.current_curve is not None:
                curve_points = self.current_curve
                if curve_points:
                    x_coords = [p[0] for p in curve_points]
                    y_coords = [p[1] for p in curve_points]
                    self.ax.plot(x_coords, y_coords, 'g-', linewidth=3)  # 线宽增加
                    
                    # 特别标记起点和终点
                    if len(curve_points) >= 2:
                        # 起点（第一个点）
                        start_x, start_y = curve_points[0]
                        self.ax.plot(start_x, start_y, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
                        self.ax.text(start_x + 15, start_y + 15, '起点', 
                                    color='red', fontsize=12, weight='bold',
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
                        
                        # 终点（最后一个点）
                        end_x, end_y = curve_points[-1]
                        self.ax.plot(end_x, end_y, 'bo', markersize=10, markeredgecolor='white', markeredgewidth=2)
                        self.ax.text(end_x + 15, end_y + 15, '终点', 
                                    color='blue', fontsize=12, weight='bold',
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        # 设置标题和坐标轴
        self.ax.set_title('光滑曲线拟合（三次样条插值）\n点击选择点，按回车保存曲线，按C清除点，按D删除点，按F完成，按Q退出')
        self.ax.axis('off')
        
        # 刷新显示
        self.fig.canvas.draw()
    
    def start_interactive_curve_fitting(self):
        """
        启动交互式曲线拟合模式
        """
        print("=" * 60)
        print("光滑曲线拟合模式（三次样条插值）")
        print("使用说明:")
        print("1. 在目标边缘点击添加点 (建议6-10个点)")
        print("2. 三个点以后，每添加一个点都会实时显示拟合的光滑曲线")
        print("3. 曲线会确保从第一个标注点开始，到最后一个标注点结束")
        print("4. 按 '回车' 键保存当前拟合的曲线")
        print("5. 按 'C' 键清除当前点集")
        print("6. 按 'D' 键删除最后一个点")
        print("7. 按 'F' 键完成所有曲线的拟合")
        print("8. 按 'Q' 键退出")
        print("注意: 曲线将确保精确通过第一个和最后一个标注点")
        print("=" * 60)
        
        # 创建图形和坐标轴
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # 连接事件
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('key_press_event', self.onkey)
        
        # 初始显示
        self.update_display()
        
        # 显示图形
        plt.tight_layout()
        plt.show()
        
        return len(self.fitted_curves) > 0

def process_all_images(source_dir, output_dir_a, output_dir_b):
    """
    处理源文件夹中的所有图像
    
    参数:
    - source_dir: 源文件夹路径
    - output_dir_a: 输出文件夹A（含背景）
    - output_dir_b: 输出文件夹B（不含背景，黑色背景）
    """
    
    # 创建输出文件夹
    Path(output_dir_a).mkdir(parents=True, exist_ok=True)
    Path(output_dir_b).mkdir(parents=True, exist_ok=True)
    
    # 获取源文件夹中的所有图像文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in os.listdir(source_dir) 
                  if os.path.splitext(f)[1].lower() in image_extensions]
    
    if not image_files:
        print(f"在 {source_dir} 中没有找到图像文件")
        return
    
    # 按文件名排序
    image_files.sort()
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    processed_count = 0
    
    for filename in image_files:
        print(f"\n处理图像: {filename}")
        print("-" * 40)
        
        # 读取图像
        image_path = os.path.join(source_dir, filename)
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"错误：无法读取图像 {filename}")
            continue
        
        # 创建交互式曲线拟合器
        fitter = InteractiveCurveFitter(img)
        
        # 运行交互式拟合
        success = fitter.start_interactive_curve_fitting()
        
        if success and fitter.fitted_curves:
            # 生成输出文件名 - 修改为PNG格式
            base_name = os.path.splitext(filename)[0]
            output_name_a = f"{base_name}_annotated.png"  # 改为PNG格式
            output_name_b = f"{base_name}_annotated.png"  # 改为PNG格式
            
            # 保存结果
            output_path_a = os.path.join(output_dir_a, output_name_a)
            output_path_b = os.path.join(output_dir_b, output_name_b)
            
            # 转换回BGR格式用于保存
            result_with_bg_bgr = cv2.cvtColor(fitter.result_with_bg, cv2.COLOR_RGB2BGR)
            
            # 对于只有曲线的图像，确保背景是黑色，线条是白色，线宽增加2个像素
            curves_only_bgr = np.zeros_like(fitter.curves_only)
            for curve in fitter.fitted_curves:
                if curve:
                    for i in range(len(curve) - 1):
                        x1, y1 = curve[i]
                        x2, y2 = curve[i+1]
                        cv2.line(curves_only_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 4)  # 线宽增加2个像素
            
            cv2.imwrite(output_path_a, result_with_bg_bgr)
            cv2.imwrite(output_path_b, curves_only_bgr)
            
            print(f"保存结果:")
            print(f"- {output_path_a} (带背景)")
            print(f"- {output_path_b} (只有白色曲线，黑色背景)")
            
            processed_count += 1
        else:
            print(f"没有为 {filename} 拟合任何曲线")
    
    print(f"\n处理完成! 成功处理 {processed_count}/{len(image_files)} 个图像")

def main():
    """
    主函数
    """
    # 设置路径（按照要求修改）
    source_directory = r"F:\wood\MVS\data\originals"
    output_directory_a = r"F:\wood\MVS\data\code\data\train\label"  # 含背景的图像
    output_directory_b = r"F:\wood\MVS\data\code\data\train\masks"  # 只有曲线的图像（黑色背景）
    
    print("开始批量处理图像...")
    print("=" * 60)
    print(f"源文件夹: {source_directory}")
    print(f"输出文件夹A (含背景): {output_directory_a}")
    print(f"输出文件夹B (只有白色曲线，黑色背景): {output_directory_b}")
    print("=" * 60)
    
    try:
        # 处理所有图像
        process_all_images(source_directory, output_directory_a, output_directory_b)
        
        print("\n所有图像处理完成!")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()