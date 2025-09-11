进入model目录:

```cmd
cd model
```

main脚本是本地测试模型,识别目录下的test_num图片

deploy脚本可使用下面命令启动MNIST的本地FastAPI接口:

```cmd
uvicorn deploy:app
```

