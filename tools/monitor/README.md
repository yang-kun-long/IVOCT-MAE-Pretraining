# IVOCT 分割训练实时监控

## 功能
- 实时显示训练指标（Dice、IoU、Sensitivity等）
- 折线图展示训练历史
- 4折交叉验证进度跟踪
- 每5秒自动刷新

## 代码位置

- 仓库源码：`tools/monitor/`
- 服务器部署：`/root/monitor`
- 监控契约：`MONITORING_CONTRACT.md`
- 数据来源：`/root/CN_seg/seven/seg/logs/progress_*.json` 和 `results_*.json`

## 部署步骤

### 1. 上传到服务器
```bash
# 在本地
cd D:\ykl\hjl\CN_seg
tar -czf monitor.tar.gz -C tools monitor
python scripts/remote_ops.py upload monitor.tar.gz /root/monitor.tar.gz
```

### 2. 在服务器上解压并安装
```bash
# SSH到服务器
ssh -p 48198 root@connect.bjb2.seetacloud.com

# 解压
cd /root
tar -xzf monitor.tar.gz
cd monitor

# 安装依赖
/root/miniconda3/bin/python -m pip install -r requirements.txt
```

### 3. 启动监控服务
```bash
# 后台运行
nohup /root/miniconda3/bin/python app.py > monitor.log 2>&1 &

# 查看日志
tail -f monitor.log
```

### 4. 配置autodl端口映射
1. 登录 https://www.autodl.com/console/instance/list
2. 找到你的实例，点击"自定义服务"
3. 添加端口映射：
   - 容器内端口：6006
   - 映射类型：HTTP
4. 保存后会得到一个公网URL，例如：https://xxxxx.autodl.pro

### 5. 访问监控界面
在浏览器打开autodl提供的URL即可实时查看训练进度！

## 本地测试
```bash
cd tools/monitor
python app.py
# 访问 http://localhost:6006
```
