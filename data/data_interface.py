#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_interface.py
@Time    :   2025/05/06 15:42:49
@Author  :   Neutrin 
'''

# here put the import lib
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib as mpl
import platform
import random
# 解决matplotlib中文显示问题
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 根据操作系统设置字体
system = platform.system()
if system == "Windows":
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
elif system == "Darwin":  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans GB', 'PingFang SC']
else:  # Linux
    # 在Linux系统中使用默认字体，避免中文字体警告
    plt.rcParams['font.family'] = 'DejaVu Sans'
    # 或者可以尝试安装的中文字体
    # plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def seed_worker(worker_id):
    # 每个worker设置独立的随机种子，保证可复现

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    # 计数器文件
    counter_file = "./runs/segmentation/worker_seeds_counter.txt"
    if not os.path.exists(os.path.dirname(counter_file)):
        os.makedirs(os.path.dirname(counter_file), exist_ok=True)
    # 读取计数器
    try:
        with open(counter_file, "r") as f:
            count = int(f.read().strip())
    except Exception:
        count = 0
    count += 1
    with open(counter_file, "w") as f:
        f.write(str(count))

    # 写入worker种子信息
    with open("./runs/segmentation/worker_seeds.txt", "a") as f:
        f.write(f"WorkerCount={count}, DataLoader worker {worker_id}: torch.initial_seed={torch.initial_seed()}, np seed={worker_seed}\n")
    print(f"[DataLoader worker {worker_id}] torch.initial_seed: {torch.initial_seed()}, np seed: {worker_seed}, WorkerCount={count}")

class SegmentationDataset(Dataset):
    """
    图像分割数据集
    """
    def __init__(self, images_dir, masks_dir, transform=None, img_size=512):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.img_size = img_size
        
        # 获取所有图像文件和路径
        self.image_files = []
        self.image_paths = []
        self.mask_paths = []
        
        # 加载图像路径
        self._load_image_paths()
        
        if len(self.image_paths) == 0:
            print(f"警告: 在 {images_dir} 目录中未找到有效的图像和掩码对")
    
    def _load_image_paths(self):
        """加载所有图像和掩码的路径 - 支持多种配对策略"""
        # 获取所有图像文件和掩码文件 - 添加.tiff支持
        img_files = [f for f in os.listdir(self.images_dir) 
                     if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
        mask_files = [f for f in os.listdir(self.masks_dir) 
                      if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
        
        print(f"找到 {len(img_files)} 个图像文件")
        print(f"找到 {len(mask_files)} 个掩码文件")
        
        # 调试：打印实际找到的文件
        if len(img_files) == 0:
            print("调试信息 - 检查图像目录内容:")
            try:
                all_files = os.listdir(self.images_dir)
                print(f"图像目录中的所有文件: {all_files[:10]}")  # 只显示前10个
                # 检查文件扩展名
                extensions = set()
                for f in all_files:
                    ext = os.path.splitext(f)[1].lower()
                    extensions.add(ext)
                print(f"发现的文件扩展名: {extensions}")
            except Exception as e:
                print(f"无法读取图像目录: {e}")
        
        # 对文件名进行排序以确保一致的配对
        img_files.sort()
        mask_files.sort()
        
        print(f"图像文件示例: {img_files[:3]}")
        print(f"掩码文件示例: {mask_files[:3]}")
        
        # 其余代码保持不变...
        
        # 取较小的数量作为配对数量
        pair_count = min(len(img_files), len(mask_files))
        
        if pair_count == 0:
            print("警告: 没有找到图像或掩码文件")
            return
        
        # 按索引配对文件（假设排序后的文件是对应的）
        for i in range(pair_count):
            img_path = os.path.join(self.images_dir, img_files[i])
            mask_path = os.path.join(self.masks_dir, mask_files[i])
            
            # 验证文件是否存在且可读
            if os.path.exists(img_path) and os.path.exists(mask_path):
                # 尝试读取文件头部验证文件完整性
                try:
                    # 简单验证：检查文件大小
                    if os.path.getsize(img_path) > 0 and os.path.getsize(mask_path) > 0:
                        self.image_files.append(img_files[i])
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)
                except Exception as e:
                    print(f"跳过损坏的文件对: {img_files[i]} - {e}")
                    continue
        
        print(f"成功配对 {len(self.image_paths)} 对图像和掩码")
        if len(self.image_paths) > 0:
            print("配对示例:")
            for i in range(min(3, len(self.image_paths))):
                print(f"  {os.path.basename(self.image_paths[i])} <-> {os.path.basename(self.mask_paths[i])}")
        else:
            print("错误: 未能创建任何有效的图像-掩码对")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        img_name = self.image_files[idx]
        
        # 读取图像和掩码
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"无法读取图像: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"无法读取掩码: {mask_path}")
        
        # 将掩码二值化（确保只有0和1）
        mask = (mask > 127).astype(np.uint8)
        
        # 应用变换
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return {
            'image': image,
            'mask': mask,
            'filename': img_name
        }


class SegmentDataInterface:
    """
    分割数据接口，处理数据集的创建和加载
    """
    def __init__(self, data_dir, batch_size=8, num_workers=4, img_size=512):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self._generator = torch.Generator()
        self._generator.manual_seed(42)
        # 设置目录结构
        self._setup_directory_structure()
        # 验证目录是否存在
        if not os.path.exists(self.train_img_dir):
            raise ValueError(f"训练数据目录不存在: {self.train_img_dir}")

        # 只创建一次 DataLoader
        self._train_loader = DataLoader(
            SegmentationDataset(
                images_dir=self.train_img_dir,
                masks_dir=self.train_mask_dir,
                transform=self.get_transforms(is_train=True),
                img_size=self.img_size
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=seed_worker,
            generator=self._generator
        )
        self._val_loader = DataLoader(
            SegmentationDataset(
                images_dir=self.val_img_dir,
                masks_dir=self.val_mask_dir,
                transform=self.get_transforms(is_train=False),
                img_size=self.img_size
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=seed_worker,
            generator=self._generator
        )
        self._test_loader = DataLoader(
            SegmentationDataset(
                images_dir=self.test_img_dir,
                masks_dir=self.test_mask_dir,
                transform=self.get_transforms(is_train=False),
                img_size=self.img_size
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=self._generator
        )

    def _setup_directory_structure(self):
        """设置数据目录结构"""
        # 基本目录
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.val_dir = os.path.join(self.data_dir, 'val')
        self.test_dir = os.path.join(self.data_dir, 'test')

        # 训练集目录结构
        self.train_img_dir = os.path.join(self.train_dir, 'img')
        self.train_mask_dir = os.path.join(self.train_dir, 'mask')

        # 验证集目录结构
        self.val_img_dir = os.path.join(self.val_dir, 'img')
        self.val_mask_dir = os.path.join(self.val_dir, 'mask')

        # 测试集目录结构
        self.test_img_dir = os.path.join(self.test_dir, 'img')
        self.test_mask_dir = os.path.join(self.test_dir, 'mask')

        # 打印目录结构
        print("数据目录结构:")
        print(f"- 训练图像: {self.train_img_dir}")
        print(f"- 训练掩码: {self.train_mask_dir}")
        print(f"- 验证图像: {self.val_img_dir}")
        print(f"- 验证掩码: {self.val_mask_dir}")
        print(f"- 测试图像: {self.test_img_dir}")
        print(f"- 测试掩码: {self.test_mask_dir}")

    def get_transforms(self, is_train=True):
        """
        获取数据增强变换
        """
        if is_train:
            return A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return self._val_loader

    def test_dataloader(self):
        return self._test_loader


def test_data_interface():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    """
    测试数据接口功能
    """
    # 测试数据路径，请根据实际情况修改
    test_data_dir = "data/DeepGlobe-Road-Extraction/split"
    
    # 如果指定的数据目录不存在，使用示例提示
    if not os.path.exists(test_data_dir):
        print(f"警告: 数据目录 '{test_data_dir}' 不存在!")
        print("请修改测试函数中的数据路径或创建正确的数据目录结构")
        return
    
    print("开始测试数据接口...")
    
    # 检查数据目录结构
    print("\n检查数据目录结构:")
    for root, dirs, files in os.walk(test_data_dir):
        level = root.replace(test_data_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        if level < 2:  # 限制递归深度，只显示前几层
            for d in dirs:
                print(f"{indent}    {d}/")
            if len(files) > 0:
                print(f"{indent}    文件: {len(files)}个")
                # 显示前几个文件名作为示例
                for i, f in enumerate(files[:3]):
                    print(f"{indent}      - {f}")
    
    # 创建数据接口实例
    data_interface = SegmentDataInterface(
        data_dir=test_data_dir,
        batch_size=4,
        num_workers=0,  # 在Windows系统调试时建议设置为0
        img_size=512
    )
    
    # 检查是否找到了图像和掩码对
    print(f"\n检查训练集目录:")
    print(f"训练图像目录: {data_interface.train_img_dir}")
    print(f"训练掩码目录: {data_interface.train_mask_dir}")
    
    if os.path.exists(data_interface.train_img_dir):
        img_files = [f for f in os.listdir(data_interface.train_img_dir) 
                     if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
        print(f"图像文件数量: {len(img_files)}")
        if len(img_files) > 0:
            print(f"示例图像文件: {img_files[:3]}")
    
    if os.path.exists(data_interface.train_mask_dir):
        mask_files = [f for f in os.listdir(data_interface.train_mask_dir) 
                      if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
        print(f"掩码文件数量: {len(mask_files)}")
        if len(mask_files) > 0:
            print(f"示例掩码文件: {mask_files[:3]}")
    
    # 获取数据加载器
    train_loader = data_interface.train_dataloader()
    val_loader = data_interface.val_dataloader()
    test_loader = data_interface.test_dataloader()
    
    # 打印数据集信息
    print(f"\n训练集大小: {len(train_loader.dataset)} 样本")
    print(f"验证集大小: {len(val_loader.dataset)} 样本")
    print(f"测试集大小: {len(test_loader.dataset)} 样本")
    
    if len(train_loader.dataset) == 0:
        print("训练集为空！请检查：")
        print("1. 图像和掩码文件是否在正确的目录中")
        print("2. 图像和掩码文件名是否匹配")
        print("3. 文件格式是否正确")
        return
    
    # 获取并显示一个批次
    print("\n获取一个批次的训练数据...")
    try:
        batch = next(iter(train_loader))
        
        # 打印批次信息
        images = batch['image']
        masks = batch['mask']
        filenames = batch['filename']
        
        print(f"批次大小: {len(images)}")
        print(f"图像张量形状: {images.shape}")
        print(f"掩码张量形状: {masks.shape}")
        print(f"掩码数值范围: min={masks.min():.3f}, max={masks.max():.3f}")
        print(f"掩码唯一值: {torch.unique(masks)}")
        print(f"文件名: {filenames}")
        
        # 可视化一个样本
        try:
            print("\n可视化一个训练样本...")
            idx = 0  # 选择第一个样本
            
            # 转换为numpy，从CHW转为HWC
            img = images[idx].numpy().transpose(1, 2, 0)
            mask = masks[idx].numpy()
            
            print(f"原始掩码形状: {mask.shape}")
            print(f"掩码数值范围: min={mask.min():.3f}, max={mask.max():.3f}")
            print(f"掩码唯一值: {np.unique(mask)}")
            
            # 反标准化图像
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            # 修复中文字体问题
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 关闭交互模式
            plt.ioff()
            
            # 创建图形
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 显示图像
            axes[0].imshow(img)
            axes[0].set_title(f"Image: {filenames[idx]}")
            axes[0].axis('off')
            
            # 显示原始掩码
            im1 = axes[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title("Mask (Gray)")
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            # 显示掩码（不同颜色映射）
            im2 = axes[2].imshow(mask, cmap='viridis')
            axes[2].set_title("Mask (Viridis)")
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            plt.savefig("data_interface_test.png", dpi=150, bbox_inches='tight')
            print(f"样本可视化已保存至: data_interface_test.png")
            
            # 同时保存原始图像和掩码用于检查
            original_img_path = train_loader.dataset.image_paths[idx]
            original_mask_path = train_loader.dataset.mask_paths[idx]
            
            print(f"原始图像路径: {original_img_path}")
            print(f"原始掩码路径: {original_mask_path}")
            
            # 读取原始文件检查
            original_img = cv2.imread(original_img_path)
            original_mask = cv2.imread(original_mask_path, cv2.IMREAD_GRAYSCALE)
            
            if original_img is not None and original_mask is not None:
                print(f"原始图像形状: {original_img.shape}")
                print(f"原始掩码形状: {original_mask.shape}")
                print(f"原始掩码数值范围: min={original_mask.min()}, max={original_mask.max()}")
                print(f"原始掩码唯一值: {np.unique(original_mask)}")
                
                # 保存原始掩码用于检查
                fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
                axes2[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
                axes2[0].set_title("Original Image")
                axes2[0].axis('off')
                
                axes2[1].imshow(original_mask, cmap='gray')
                axes2[1].set_title("Original Mask")
                axes2[1].axis('off')
                
                plt.tight_layout()
                plt.savefig("original_data_check.png", dpi=150, bbox_inches='tight')
                print("原始数据检查图像已保存至: original_data_check.png")
            
            plt.close('all')
            
        except Exception as e:
            print(f"可视化过程出错: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"获取训练批次时出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n数据接口测试完成!")


if __name__ == "__main__":
    test_data_interface()