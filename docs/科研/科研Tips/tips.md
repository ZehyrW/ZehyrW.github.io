

### 1.针对要从huggingface上下载模型和数据集如何解决？

镜像网站 + huggingface-cli

> 会将下载的模型和数据集自动链接到默认的目录下

```bash
# 首先，pip安装库 huggingface-cli
 pip install -U huggingface_hub
# 然后，设置镜像地址
 export HF_ENDPOINT=https://hf-mirror.com
 
# 模型下载指令
huggingface-cli download --resume-download gpt2 --local-dir gpt2
# 数据集下载指令
huggingface-cli download --repo-type dataset --resume-download wikitext --local-dir wikitext
```

> local-dir填写本地地址即可



### 2.使用Jupyter运行程序释放内存指令

```bash
ps -ef | grep jupyter | grep -v grep | awk '{print $2}' | xargs kill -9
```



### 3. 使用transformers库调用模型进行单机多卡训练[reference]([处理大型模型进行推理 (huggingface.co)](https://huggingface.co/docs/accelerate/usage_guides/big_modeling))

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3'

from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp", device_map="auto", torch_dtype=torch.float16)
```

> * device_map = ‘auto’ 用来自动使用定义的cuda设别 （这里似乎只能指定3个设备 ≥3时会出现tensor存在于两个device的情况）
>
> * torch_dtype = torch.float16 通过参数指定模型加载到的精度来节省内存



### 4. tqdm在一行内显示

```python
import time
from tqdm import range
# 自动更新
for i in tqdm(range(10)): # 共可以更新10次进度条
    time. Sleep(0.5) # 每次更新间隔0.5s
```



### 5. 远程服务器免密登录

```bash
# 首先在本机上生成一对公钥和私钥（win+R进入cmd）
ssh-keygen -t rsa
# 一直回车会在./ssh文件得到两个文件
./ssh/id_rsa ==>私钥
./ssh/id_rsa.pub ==>公钥
```

```bash
# 然后将公钥上传到远程服务器上
# 方法一
ssh-copy-id -i ~/.ssh/id_rsa.pub -p port 'username@ip'

# 方法二
scp ~/.ssh/id_rsa.pub -p port username@ip:~/home(主机终端)
cat ~/home/id_rsa.pub >> ~/.ssh/authorizes_keys(远程服务器终端)
```

> powershell无法使用ssh-copy-id 怎么办？
>
> * ```bash
>   function ssh-copy-id([string]$userAtMachine, $args){   
>       $publicKey = "$ENV:USERPROFILE" + "/.ssh/id_rsa.pub"
>       if (!(Test-Path "$publicKey")){
>           Write-Error "ERROR: failed to open ID file '$publicKey': No such file"            
>       }
>       else {
>           & cat "$publicKey" | ssh $args $userAtMachine "umask 077; test -d .ssh || mkdir .ssh ; cat >> .ssh/authorized_keys || exit 1"      
>       }
>   }
>   ```
>
> 复制上面脚本，粘贴到终端，回车运行
>
> * 通过git-bash实现（**最简单的方式**）
> * 暴力复制粘贴😂



### 6. 网页部署

```bash
# 第1步 编译React项目 得到build文件夹
npm run build
# 第2步 安装nginx服务器软件
ssh 服务器用户@公网ip
yum install epel-release -y 
yum install nginx -y
​```
开启防火墙：systemctl start nginx
启动nginx：service nginx start
重启nginx：nginx -s reload
关闭nginx：service nginx stop
​```
服务器输入公网ip查看是否启动成功
# 第3步 上传编译得到的build文件夹到云服务器
eg. /www/sed
# 第4步 在/etc/nginx/conf.d文件夹下创建配置文件 eg. sed.conf(前缀自定义)
​```
server {
    listen      80; #端口号
    server_name 47.113.223.186; #域名
    root /www/sed; #文件路径
    index index.html; #配置默认访问的界面
    #输入错误路径 跳转到index.html
    location / {
        try_files $uri $uri/    /index.html;
    }
}
​```
# 第5步 重启nginx服务器 服务器输入公网ip查看是否成功
```

