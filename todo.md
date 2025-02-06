# 模型搭建

- [x] 模型搭建
- [ ] viz
  - [x] loss
- [x] 推理

# Cstimer 接入，用于收集人类偏好

- [x] 计时结束，设置label功能
    - [x] 主界面button
    - [x] 键盘shortcut
- [x] 导出数据
    - [x] 数据存储
    - [x] 导出界面button，直接使用现有的导出文件button

# 推理接入

- [x] 导出时魔方种类+打乱 -> 状态，py实现
- [x] onnx export
- [x] onnx rt + wasm部署cstimer
- [ ] 重构，不使用单独的tool，融合进打乱图案功能
- [ ] 支持本地上传 onnx 模型
- [ ] 在线训练？


# 训练相关

- [x] 正阶魔方支持
- [x] 代码重构，使用data loader
- [x] 数据增强
  - [x] 魔表。交换两面打乱
  - [x] 正阶魔方。~~颜色交换~~ use pre scramble to do it
- [ ] 无data files设置时的支持