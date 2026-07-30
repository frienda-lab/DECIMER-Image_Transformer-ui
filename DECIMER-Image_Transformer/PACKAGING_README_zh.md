# DECIMER Desktop 安装版

构建产物位于 `release/DECIMER-Desktop-Setup.exe`。

安装程序将应用解压到当前用户的：

`%LOCALAPPDATA%\Programs\DECIMER Desktop`

安装结束后会创建桌面快捷方式并启动应用。应用包含 Python、TensorFlow、DECIMER 普通识别模型、手绘识别模型和分割模型，不需要另行配置环境，也不需要首次联网下载模型。

未进行代码签名，因此 Windows SmartScreen 可能显示“未知发布者”。这是本地构建的软件，可选择“更多信息 → 仍要运行”。
