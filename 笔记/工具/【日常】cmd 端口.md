**1. windows下查看端口使用情况**

`netstat -aon` 

查找具体端口，以查找9000为例：

`netstat -aon|findstr "8080"` 

会显示出占用端口进程的`pid`信息。

**2. kill某个进程**

`taskkill /f /pid 9523`

