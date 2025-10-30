# GitHub使用指南

## 初始化仓库：
```bash
git init  
git add .    
git add . ':!*.gitignore'  #使用gitignore
git commit -m "Initial commit"  
```

### gitgnore写法：  
```nano .gitignore```   
```bash
万一在使用之前就使用了相关文件，按照下面方法删除仓库里的文件。  
git rm -r --cached data left-eye preTrainedCheckpoints pretrainCheckpoint checkpoint runs  
git add .gitignore  
git commit -m "Update .gitignore and remove large files from tracking"  
```

## 配置仓库
```bash
git remote add origin git@github.com:你的用户名/仓库名.git  
git push -u origin master  
git push  #后面都可以使用这个
```
## 查看仓库
```git remote -v```
