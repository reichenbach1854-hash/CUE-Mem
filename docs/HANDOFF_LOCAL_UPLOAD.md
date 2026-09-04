# CUE-Mem 静态 Demo 本地上传 Handoff

## 任务目标

请将 CUE-Mem 静态展示版 demo 上传到 GitHub 仓库：

```text
https://github.com/reichenbach1854-hash/CUE-Mem
```

上传后使用 GitHub Pages 发布网站。

## 静态 Demo 文件

静态 demo 发布目录为：

```text
docs/
```

该目录是可以直接发布的静态网站，包含：

- `index.html`
- `static/app.js`
- `static/style.css`
- `static/figures/rq2_caption_quality.png`
- `data/` 下的流程、QA 和实验结果 JSON
- `media/` 下的图片和音频
- `.nojekyll`

该目录约 40 MB，只包含 demo 实际使用的资源，不要上传整个 CUE-Mem 项目。

## 本地操作步骤

### 1. 克隆 GitHub 仓库

如果本地还没有仓库：

```bash
git clone https://github.com/reichenbach1854-hash/CUE-Mem.git
cd CUE-Mem
```

如果本地已经有该仓库：

```bash
cd /本地路径/CUE-Mem
git status
git pull origin main
```

提交前必须先确认本地没有未保存的重要修改。

### 2. 检查发布目录

```bash
find docs -maxdepth 2 -type f | sort | head -30
du -sh docs
test -f docs/index.html && echo "index.html exists"
test -f docs/.nojekyll && echo ".nojekyll exists"
```

不要把模型权重、完整原始数据或其他大文件复制到 `docs/`。

### 3. 提交并推送

```bash
git status --short
git add docs
git commit -m "Add static CUE-Mem benchmark demo"
git push origin main
```

不要执行：

```bash
git add .
```

这样可以避免误提交仓库中的其他文件或大模型文件。

如果本地使用 SSH remote，也可以使用 SSH 推送；如果使用 HTTPS，GitHub 密码位置需要填写 Personal Access Token，而不是 GitHub 登录密码。

## GitHub Pages 配置

推送成功后，进入 GitHub 仓库：

```text
Settings
→ Pages
→ Build and deployment
→ Source: Deploy from a branch
→ Branch: main
→ Folder: /docs
→ Save
```

网站地址预计为：

```text
https://reichenbach1854-hash.github.io/CUE-Mem/
```

首次部署可能需要等待一段时间。

## 功能说明

该静态 demo 不需要运行 Python 服务，浏览器会读取 `docs/data/` 下的 JSON 文件，并加载 `docs/media/` 下的图片和音频。

已保留的功能包括：

- 数据构造流程逐步展开
- 自动滚动、自动播放、上一步、下一步、重置和回到上方
- 四类 QA 案例
- 图片和音频展示
- RQ1、RQ2、RQ3 实验结果
- RQ2 论文图
- RQ3 四种 Index/Use 变体
- 中英文切换
- 柱状图和实验结论

## 排错提示

如果页面能打开但显示空白或数据读取失败，检查：

1. GitHub Pages 是否设置为 `main` 分支的 `/docs` 目录；
2. `docs/data/` 是否完整上传；
3. `docs/media/` 是否完整上传；
4. 浏览器访问地址是否包含 `/CUE-Mem/`；
5. 是否使用了最新提交并进行强制刷新。

如果页面使用旧版本资源，可以执行浏览器强制刷新：

```text
Ctrl + Shift + R
```

## 当前不要做的事情

- 不要把整个 `/share/home/ylhu/zmlong/CUE-Mem/` 上传到 GitHub Pages；
- 不要把模型权重上传到 `docs/`；
- 不要删除 `docs/` 或本地提交；
- 不要修改静态 demo 的绝对/相对资源路径，除非已经验证 GitHub Pages 项目路径。
