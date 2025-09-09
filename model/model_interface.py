#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model_interface.py
@Time    :   2025/05/07 14:36:41
@Author  :   Neutrin 
'''

"""
model_interface.py - 基础深度卷积神经网络模块

功能：
提供用于图像分割的多种神经网络接口
"""

from sympy import Ei
import torch
import torch.nn as nn
from typing import Optional, Tuple
import os
import sys
import time
import warnings
# 忽略警告
warnings.filterwarnings("ignore")

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入模型 - 改为绝对导入
try:
    from PSSNF.model.基准模型.UNet import UNet
    from PSSNF.model.models.EMCAD import EMCADNet
    from PSSNF.model.models.R2Unet import R2U_Net
    from PSSNF.model.models.AttUnet import AttU_Net
    from PSSNF.model.models.R2AttUNet import R2AttU_Net
    from PSSNF.model.models.NestedUNet import NestedUNet
    from PSSNF.model.models.DictUNet import Unet_dict
    from PSSNF.model.models.SegNet import SegNet
    from PSSNF.model.models.EffUNet import EffUNet
    from PSSNF.model.models.CBAMUNet import CBAMUNet
    from PSSNF.model.models.SEUNet import SEUNet
    from PSSNF.model.models.DEGANet import DEGANet  
    # from PSSNF.model.FDNet import FDNet, create_fdnet
    from PSSNF.model.Net import DemoNet  # demo
    from PSSNF.model.基准模型.DF4B import DF4B
    
except ImportError as e:
    print(f"导入错误: {e}")
    print("尝试本地导入...")
    try:
        from PSSNF.model.基准模型.UNet import UNet
        from models.EMCAD import EMCADNet
        from models.R2Unet import R2U_Net
        from models.AttUnet import AttU_Net
        from models.R2AttUNet import R2AttU_Net
        from models.NestedUNet import NestedUNet
        from models.DictUNet import Unet_dict
        from models.SegNet import SegNet
        from models.EffUNet import EffUNet
        from models.CBAMUNet import CBAMUNet
        from models.SEUNet import SEUNet
        from models.DEGANet import DEGANet
        # from FDNet import FDNet, create_fdnet
        from PSSNF.model.Net import DemoNet  # demo
        from PSSNF.model.基准模型.DF4B import DF4B
        
    except ImportError as e2:
        print(f"本地导入也失败: {e2}")
        sys.exit(1)

# 导入GFLOPS计算工具
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False


class MInterface:
    """模型接口类，用于创建、加载、保存不同类型的分割模型"""
    
    def __init__(self, 
                 model_type: str = "UNet",
                 in_channels: int = 3, 
                 num_classes: int = 1, 
                 device: Optional[torch.device] = None,
                 **kwargs):
        """
        初始化模型接口
        
        Args:
            model_type: 模型类型（"UNet", "EMCADNet", "R2U_Net"）
            in_channels: 输入通道数
            num_classes: 分类类别数（分割掩码的通道数）
            device: 模型设备
            kwargs: 传递给具体模型的其他参数
        """
        self.model_type = model_type
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.kwargs = kwargs
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # 创建模型
        self.model, self.param_count = self._create_model()
        
    def _create_model(self) -> Tuple[nn.Module, int]:
        """根据模型类型创建模型实例"""
        if self.model_type == "UNet":
            model = UNet(
                n_channels=self.in_channels,
                n_classes=self.num_classes
            )
        elif self.model_type == "EMCADNet":
            model = EMCADNet(
                in_channels=self.in_channels,
                num_classes=self.num_classes,
                **self.kwargs
            )
        elif self.model_type == "R2UNet":
            model = R2U_Net(
                img_ch=self.in_channels,
                output_ch=self.num_classes,
                t=self.kwargs.get('t', 2)
            )
        elif self.model_type == "AttUNet":
            model = AttU_Net(
                img_ch=self.in_channels,
                output_ch=self.num_classes
            )
        elif self.model_type == "R2AttUNet":
            model = R2AttU_Net(
                img_ch=self.in_channels,
                output_ch=self.num_classes,
                t=self.kwargs.get('t', 2)
            )
        elif self.model_type == "NestedUNet":
            model = NestedUNet(
                img_ch=self.in_channels,
                output_ch=self.num_classes
            )
        elif self.model_type == "DictUNet":
            model = Unet_dict(
                n_labels=2,        
                n_filters=64,      
                p_dropout=0.3,     
                batchnorm=True     
            )
        elif self.model_type == "SegNet":
            model = SegNet(
                in_chn=self.in_channels,
                out_chn=self.num_classes,
                BN_momentum=0.5
            )
        elif self.model_type == "EffUNet":
            model = EffUNet(
                in_channels=self.in_channels,
                classes=self.num_classes,
            )
        elif self.model_type == "CBAMUNet":
            model = CBAMUNet(
                img_ch=self.in_channels,
                output_ch=self.num_classes,
                **self.kwargs
            )    
        elif self.model_type == "SEUNet":
            model = SEUNet(
                img_ch=self.in_channels,
                output_ch=self.num_classes,
                enhanced=self.kwargs.get('enhanced', False),
                reduction=self.kwargs.get('reduction', 16)
            )
        elif self.model_type == "DEGANet":
            model = DEGANet(
                in_channels=self.in_channels,
                out_channels=self.num_classes,
                **self.kwargs
            )
        elif self.model_type == "DemoNet":
            model = DemoNet(
                in_channels=3, 
                num_classes=1
            )
        elif self.model_type == "DF4B":
            model = DF4B(
                in_channels=self.in_channels,
                num_classes=self.num_classes
            )
        elif self.model_type == "DF4C":
            from PSSNF.model.基准模型.DF4C import DF4C
            model = DF4C(
                in_channels=self.in_channels,
                num_classes=self.num_classes
            )
        elif self.model_type == "DF4D":
            from PSSNF.model.基准模型.DF4D import DF4D
            model = DF4D(
                in_channels=self.in_channels,
                num_classes=self.num_classes
            )
        elif self.model_type == "DF4E":
            from PSSNF.model.基准模型.DF4E import DF4E
            model = DF4E(
                in_channels=self.in_channels,
                num_classes=self.num_classes
            )
        elif self.model_type == "DF4F":
            from PSSNF.model.基准模型.DF4F import DF4F
            model = DF4F(
                in_channels=self.in_channels,
                num_classes=self.num_classes
            )
        elif self.model_type == "HXD1":
            from PSSNF.model.基准模型.HXD1 import HXD1
            model = HXD1(
                in_channels=self.in_channels,
                num_classes=self.num_classes
            )
        elif self.model_type == "HXD1B":
            from PSSNF.model.基准模型.HXD1B import HXD1B
            model = HXD1B(
                in_channels=self.in_channels,
                num_classes=self.num_classes,
            )
        elif self.model_type == "HXD1BA":
            from PSSNF.model.基准模型.HXD1BA import HXD1BA
            model = HXD1BA(
                in_channels=self.in_channels,
                num_classes=self.num_classes
            )
        elif self.model_type == "HXD1C":
            from PSSNF.model.基准模型.HXD1C import HXD1C
            model = HXD1C(
                in_channels=self.in_channels,
                num_classes=self.num_classes
            )
        elif self.model_type == "HXD1CA":
            from PSSNF.model.基准模型.HXD1CA import HXD1CA
            model = HXD1CA(
                in_channels=self.in_channels,
                num_classes=self.num_classes
            )
        elif self.model_type == "HXD2":
            from PSSNF.model.基准模型.HXD2 import HXD2
            model = HXD2(
                in_channels=self.in_channels,
                num_classes=self.num_classes
            )
        elif self.model_type == "ULSCSS2":
            from PSSNF.model.基准模型.ULSCSS2 import ULSCSS2
            model = ULSCSS2(
                in_channels=self.in_channels,
                num_classes=self.num_classes
            )
        elif self.model_type == "CR200":
            from PSSNF.model.基准模型.CR200 import CR200
            model = CR200(
                in_channels=3,
                num_classes=1
            )
        elif self.model_type == "FXD1":
            from PSSNF.model.基准模型.FXD1 import FXD1
            model = FXD1(
                in_channels=self.in_channels,
                num_classes=self.num_classes
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
            
        # 移动模型到指定设备
        model = model.to(self.device)
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        
        return model, total_params
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行前向传播"""
        return self.model(x)
    
    def calculate_fps(self, input_size=(1, 3, 512, 512), num_runs=100, warmup_runs=10):
        """
        计算模型的FPS（每秒处理帧数）
        
        Args:
            input_size: 输入张量大小
            num_runs: 测试运行次数
            warmup_runs: 预热运行次数
            
        Returns:
            tuple: (平均FPS, 平均推理时间)
        """
        try:
            self.model.eval()
            test_input = torch.randn(input_size).to(self.device)
            
            # 预热运行
            with torch.no_grad():
                for _ in range(warmup_runs):
                    _ = self.model(test_input)
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
            
            # 正式测试
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = self.model(test_input)
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
            
            end_time = time.time()
            total_time = end_time - start_time
            avg_time_per_frame = total_time / num_runs
            fps = 1.0 / avg_time_per_frame
            
            return fps, avg_time_per_frame * 1000  # 返回毫秒
            
        except Exception as e:
            return f"FPS计算失败: {str(e)}", "N/A"
    
    def calculate_gflops(self, input_size=(1, 3, 512, 512)):
        """计算模型的GFLOPS"""
        if not THOP_AVAILABLE:
            return "thop库未安装", "N/A"
        
        try:
            test_input = torch.randn(input_size).to(self.device)
            flops, params = profile(self.model, inputs=(test_input,), verbose=False)
            flops, params = clever_format([flops, params], "%.3f")
            return flops, params
        except Exception as e:
            return f"计算失败: {str(e)}", "N/A"
    
    def save_model(self, path: str) -> None:
        """保存模型权重"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"模型已保存到: {path}")
        
    def load_model(self, path: str) -> None:
        """加载模型权重"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"已从 {path} 加载模型")