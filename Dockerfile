# 使用官方 Python 3.9 镜像作为基础镜像
FROM python:3.9

# 设置工作目录
WORKDIR /app

# 复制当前目录下的所有文件到工作目录
COPY . /app

# 安装依赖
RUN pip install -r requirements.txt

# 容器启动时运行的命令
CMD ["python", "./run.py"]
