#!/bin/bash

# 挥手动作训练快速启动脚本
# 使用优化后的配置进行训练

echo "🚀 开始挥手动作训练..."
echo "📊 配置信息："
echo "   - 任务: hi_mimic"
echo "   - 环境数量: 4096"
echo "   - 设备: cuda:0"
echo "   - 动作文件: waving.json"
echo "   - 模式: 无头模式"
echo ""

# 检查是否在正确的目录
if [ ! -f "mimic_real/scripts/train.py" ]; then
    echo "❌ 错误: 请确保在DeepMimic_hi目录下运行此脚本"
    exit 1
fi

# 检查waving.json文件是否存在
if [ ! -f "mimic_real/data/hi/waving.json" ]; then
    echo "❌ 错误: waving.json文件不存在"
    echo "请确保文件位于: mimic_real/data/hi/waving.json"
    exit 1
fi

echo "✅ 文件检查通过"
echo ""

# 创建logs目录（如果不存在）
mkdir -p logs

# 启动训练
echo "🎯 启动训练..."
python mimic_real/scripts/train.py \
    --task=hi_mimic \
    --num_envs=4096 \
    --headless \
    --device=cuda:0

echo ""
echo "🎉 训练完成！"
echo "📁 查看训练日志："
echo "   tensorboard --logdir logs/"
echo ""
echo "🧪 测试训练结果："
echo "   python mimic_real/scripts/play.py --task=hi_mimic --num_envs=2 --device=cuda:0" 