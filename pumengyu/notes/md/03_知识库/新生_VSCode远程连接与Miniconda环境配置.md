# 新生入门：用 VS Code 连接服务器并配置 Miniconda 环境




## 2. 在本地电脑安装 VS Code

官方下载页：<https://code.visualstudio.com/Download>

### Windows

1. 打开下载页，选择 Windows 的 **User Installer x64**。
2. 运行下载得到的 `.exe` 安装程序。
3. 建议勾选“添加到 PATH”和“添加到右键菜单”（如果安装器提供这些选项）。
4. 安装完成后启动 VS Code。

### macOS

1. 点击左上角苹果菜单 → **关于本机**，查看芯片类型：
   - 显示 Apple M1、M2、M3、M4 等：选择 **Apple silicon**；
   - 显示 Intel：选择 **Intel chip**。
2. 从官方下载页下载对应的 `.zip`。
3. 解压后，把 `Visual Studio Code.app` 拖入“应用程序”文件夹并打开。

## 3. 安装 Remote - SSH 扩展

1. 在 VS Code 左侧点击“扩展”图标，或按：
   - Windows：`Ctrl+Shift+X`
   - macOS：`Command+Shift+X`
2. 搜索 **Remote - SSH**。
3. 确认发布者是 **Microsoft**，点击 **Install**。

官方说明：<https://code.visualstudio.com/docs/remote/ssh>

## 4. 先在系统终端测试 SSH

先不要急着用 VS Code。确认系统自带的 SSH 能连接，之后排错会更容易。

### Windows

打开 PowerShell（开始菜单搜索 `PowerShell`），先检查：

```powershell
ssh -V
```

### macOS

打开“终端”（Terminal），先检查：

```bash
ssh -V
```

### 连接服务器

Windows 和 macOS 使用同样的命令。把尖括号中的内容替换成管理员提供的信息，输入时不要保留 `<` 和 `>`：

```bash
ssh -p <端口> <用户名>@<服务器地址>
```

例如：

```bash
ssh -p 22 zhangsan@10.0.0.10
```

第一次连接可能出现主机指纹确认。先核对服务器地址无误，再输入 `yes`。随后按提示输入密码；输入密码时屏幕**不会显示字符或星号**，这是正常现象，输完直接回车。

登录成功后可执行：

```bash
hostname
whoami
pwd
```

它们应分别显示服务器主机名、你自己的用户名和当前远程目录。然后输入 `exit` 退出测试连接。

> 如果出现 `Connection timed out`、`Connection refused` 或密码反复失败，先截图完整报错并联系管理员。不要连续猜密码。

## 5. 用 VS Code 连接服务器

1. 打开 VS Code。
2. 按 `F1`，或按：
   - Windows：`Ctrl+Shift+P`
   - macOS：`Command+Shift+P`
3. 输入并选择 **Remote-SSH: Add New SSH Host...**。
4. 输入刚才测试成功的命令，例如：

   ```bash
   ssh -p 22 zhangsan@10.0.0.10
   ```

5. 选择默认的 SSH 配置文件。
6. 再次打开命令面板，选择 **Remote-SSH: Connect to Host...**，然后选择刚添加的服务器。
7. 如果询问服务器类型，选择 **Linux**。
8. 输入密码，等待 VS Code 自动配置远程组件。

连接成功后，VS Code 左下角会显示类似 `SSH: 服务器名称`。选择 **File → Open Folder...**，打开管理员提供的项目目录。

### 推荐的 SSH 配置写法

需要检查或修改配置时，在 VS Code 命令面板中运行 **Remote-SSH: Open SSH Configuration File...**。配置示例：

```sshconfig
Host lab-server
    HostName 10.0.0.10
    User zhangsan
    Port 22
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

保存后，连接列表中会出现 `lab-server`。`HostName`、`User` 和 `Port` 必须替换成自己的信息。

常见的本地配置文件位置：

- Windows：`C:\Users\你的Windows用户名\.ssh\config`
- macOS：`~/.ssh/config`

## 6. 在服务器安装 Miniconda 并创建环境

下面的命令都要在**已经连接服务器的 VS Code 终端**中执行。打开方式：**Terminal → New Terminal**。

### 6.1 确认自己位于服务器

```bash
hostname
whoami
pwd
uname -m
```

`uname -m` 常见输出：

- `x86_64`：大多数 Intel/AMD Linux 服务器；
- `aarch64`：ARM Linux 服务器。

### 6.2 下载对应的 Miniconda 安装包

Miniconda 官方安装说明：<https://www.anaconda.com/docs/getting-started/miniconda/install>

如果 `uname -m` 输出 `x86_64`，执行：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

如果输出 `aarch64`，执行：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
```

如果服务器没有 `wget`，可改用 `curl`，例如 x86_64 服务器：

```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

### 6.3 安装到自己的用户目录

以下以常见的 `x86_64` 为例：

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

安装过程中的选择：

1. 按回车继续阅读许可协议，按空格翻页；
2. 输入 `yes` 接受协议；
3. 安装路径使用默认的个人目录（通常是 `/home/你的用户名/miniconda3`）；
4. 当询问是否运行 `conda init` 时输入 `yes`。

如果安装的是 ARM 版本，请把命令中的文件名换成 `Miniconda3-latest-Linux-aarch64.sh`。

安装结束后，关闭当前终端并在 VS Code 中新建一个终端，然后检查：

```bash
conda --version
```

如果仍提示 `conda: command not found`，执行：

```bash
source ~/miniconda3/bin/activate
conda init bash
```

再关闭并新建终端。

不希望每次打开终端都自动进入 `(base)` 时，可执行：

```bash
conda config --set auto_activate_base false
```

### 6.4 创建自己的虚拟环境

下面以环境名 `medseg`、Python 3.10 为例。若项目负责人指定了其他名称或 Python 版本，以项目要求为准。

```bash
conda create -n medseg python=3.10 -y
conda activate medseg
```

验证当前环境：

```bash
which python
python --version
conda env list
```

`which python` 的结果应类似：

```text
/home/你的用户名/miniconda3/envs/medseg/bin/python
```

以后每次进入服务器、开始项目工作前，先运行：

```bash
conda activate medseg
```

退出当前环境：

```bash
conda deactivate
```

> 不要直接照搬网上的 CUDA、PyTorch 安装命令。它们需要和服务器驱动、项目代码及老师指定版本匹配，收到项目环境要求后再安装。

## 7. 可选：在本地电脑安装 Miniconda

如果你只通过 VS Code 使用服务器，本地电脑通常不必安装 Miniconda。只有需要在本地运行 Python 时才安装。

Miniconda 官方下载与说明：

- 下载入口：<https://www.anaconda.com/download>
- 安装说明：<https://www.anaconda.com/docs/getting-started/miniconda/install>
- 官方安装包目录：<https://repo.anaconda.com/miniconda/>

请确认产品名称是 **Miniconda**，不要下载完整的 **Anaconda Distribution**。

### Windows 本地安装

1. 选择 **Miniconda Windows 64-bit Graphical Installer**（`.exe`）。
2. 安装范围选择 **Just Me**，不要以管理员身份安装。
3. 建议使用无空格、无中文和无特殊字符的安装路径。
4. 官方不建议勾选 **Add Miniconda3 to my PATH**；安装后从开始菜单打开 **Anaconda Prompt (Miniconda3)** 使用 `conda`。
5. 在 Anaconda Prompt 中运行 `conda --version` 验证。

### macOS 本地安装

先在“关于本机”确认芯片：

- Apple M 系列芯片：下载 `MacOSX-arm64` 安装包；
- Intel 芯片：下载 `MacOSX-x86_64` 安装包。

新生建议选择对应架构的图形安装器（`.pkg`），按安装器提示完成后，重新打开终端并运行：

```bash
conda --version
```

## 8. 在 VS Code 中选择服务器上的 Python 环境

1. 确认左下角显示 `SSH: ...`，即当前处于远程窗口。
2. 在扩展面板搜索 **Python**，确认发布者是 **Microsoft**，并点击 **Install in SSH: ...**。
3. 按 `F1`，选择 **Python: Select Interpreter**。
4. 选择路径中含有 `/miniconda3/envs/medseg/bin/python` 的解释器。
5. 新建终端，执行 `which python` 和 `python --version` 再次确认。

注意：扩展面板会区分“本地安装”和“安装在 SSH 服务器”。Python 代码要在服务器运行时，应确保 Python 扩展也已安装到对应的远程端。

## 9. 常见问题

### `ssh` 不是内部或外部命令（Windows）

在 Windows 设置中搜索“可选功能”，安装 **OpenSSH Client**，重开 PowerShell 后再运行 `ssh -V`。

### `Permission denied`

通常是用户名、密码、端口或密钥不正确。核对管理员提供的信息，不要反复猜密码。

### `Connection timed out` 或一直卡住

检查是否需要校园网或实验室 VPN，并确认服务器地址和端口。把完整错误信息发给管理员。

### VS Code 一直停在 Installing VS Code Server

先确认系统终端中的 `ssh` 可以正常登录，再在 VS Code 中打开 **View → Output**，选择 **Remote - SSH** 查看日志并将完整日志交给管理员。

### `conda activate` 报错

在服务器终端执行：

```bash
source ~/miniconda3/bin/activate
conda init bash
```

关闭终端并新建一个终端后重试。

### 安装包选错架构

服务器安装包由服务器的 `uname -m` 决定，与学生使用 Windows 还是 Mac 无关。Mac 本地安装包才由 Mac 的 Apple/Intel 芯片决定。

## 10. 完成检查表

请逐项确认：

- [ ] 本地电脑已安装 VS Code；
- [ ] 已安装 Microsoft 发布的 Remote - SSH 扩展；
- [ ] 在 PowerShell 或 macOS 终端中可以用 `ssh` 登录服务器；
- [ ] VS Code 左下角显示 `SSH: ...`；
- [ ] VS Code 可以打开远程项目目录；
- [ ] 服务器终端运行 `conda --version` 有正常输出；
- [ ] 已创建并激活自己的项目环境；
- [ ] `which python` 指向自己的 `miniconda3/envs/...`；
- [ ] VS Code 已选择服务器上对应的 Python 解释器；
- [ ] 没有把密码或 SSH 私钥发给其他人。

完成后，把下面四条命令的输出截图交给指导人员即可（截图前确认其中没有密码、令牌等敏感信息）：

```bash
hostname
whoami
conda --version
which python
```

