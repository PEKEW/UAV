#!/usr/bin/env python3
"""
修复数据集比例问题的脚本
"""
import os
import subprocess
from pathlib import Path

def run_command(command, description):
    """运行命令并显示结果"""
    print(f"\n{'='*50}")
    print(f"执行: {description}")
    print(f"命令: {command}")
    print(f"{'='*50}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print("输出:")
        print(result.stdout)
    
    if result.stderr:
        print("错误:")
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"命令执行失败，返回码: {result.returncode}")
        return False
    
    return True

def main():
    """主函数"""
    print("开始修复数据集比例问题...")
    
    # 1. 重新生成EVTOL数据集
    os.chdir("evtol_dataset")
    success = run_command("python create_evtol_dataset.py", "重新生成EVTOL数据集")
    if not success:
        print("EVTOL数据集生成失败")
        return
    
    # 2. 重新生成飞行数据集
    os.chdir("../flight_dataset")
    success = run_command("python create_flight_dataset.py", "重新生成飞行数据集")
    if not success:
        print("飞行数据集生成失败")
        return
    
    # 3. 运行测试验证
    os.chdir("../analysis")
    success = run_command("python test_datasets.py", "运行数据集测试")
    if not success:
        print("数据集测试失败")
        return
    
    print("\n" + "="*60)
    print("数据集修复完成！")
    print("建议运行可视化检查:")
    print("cd analysis && python visualize_datasets.py")
    print("="*60)

if __name__ == "__main__":
    # 确保在comp目录下运行
    if not Path("evtol_dataset").exists() or not Path("flight_dataset").exists():
        print("请在comp目录下运行此脚本")
        exit(1)
    
    main()