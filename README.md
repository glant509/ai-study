# AI Agent 工程学习手册

一个可直接部署到 GitHub Pages 的 VitePress + Markdown + PWA 中文学习站点。包含响应式 PC / 手机布局、本地全文搜索、Mermaid、代码高亮、离线缓存与自动部署。

## 本地开发

需要 Node.js 20 或更高版本。

```bash
npm install
npm run docs:dev
```

浏览器打开终端显示的地址。生产构建与本地预览：

```bash
npm run docs:build
npm run docs:preview
```

构建产物位于 `docs/.vitepress/dist/`。

## 内容结构

课程位于 `docs/`，按 `llm/`、`agent/`、`rag/`、`mcp/`、`harness/`、`infra/`、`security/`、`roadmap/` 分组。新增 Markdown 后，在 `docs/.vitepress/config.mts` 的 `sidebar` 中加入链接。

Mermaid 图使用围栏代码块：

````markdown
```mermaid
flowchart LR
  A --> B
```
````

## GitHub Pages 部署

1. 新建 GitHub 仓库，例如 `ai-agent-handbook`。
2. 将本项目提交并推送到 `main` 分支。
3. 打开仓库 **Settings → Pages**，在 **Build and deployment** 中选择 **GitHub Actions**。
4. 推送后 `.github/workflows/deploy.yml` 会自动构建和发布。

普通项目仓库的访问地址通常是：

```text
https://<用户名>.github.io/<仓库名>/
```

工作流会读取真实仓库名并自动设置 base path，所以仓库改名后再次部署即可，无需手改代码。

## 修改仓库名或 base

本地默认仓库名是 `ai-agent-handbook`。换名后可这样开发：

```bash
VITEPRESS_REPO_NAME=my-new-repo npm run docs:dev
```

也可以复制 `.env.example` 为 `.env.local` 并修改值。若仓库本身叫 `<用户名>.github.io`，或使用自定义域名，base 应为 `/`：

```bash
VITEPRESS_REPO_NAME= VITEPRESS_BASE=/ npm run docs:build
```

如需完全指定路径，设置 `VITEPRESS_BASE=/some-path/`。base 必须以 `/` 开头和结尾。

## 绑定自定义域名

1. 在 `docs/public/CNAME` 新建一行，写入域名，例如 `agent.example.com`。
2. 在 DNS 服务商添加 CNAME：`agent` 指向 `<用户名>.github.io`。
3. 在 GitHub **Settings → Pages → Custom domain** 填写该域名并启用 HTTPS。
4. 修改工作流 Build 步骤：把 `VITEPRESS_REPO_NAME` 设为空、`VITEPRESS_BASE` 设为 `/`，并将 `VITEPRESS_SITE_URL` 设为完整自定义域名。

> GitHub Pages 的 DNS 记录与界面可能调整，绑定时以 GitHub Pages 设置页给出的校验结果为准。

## PWA 与离线使用

部署后首次联网打开站点，浏览器会注册 Service Worker 并缓存站点资源。iPhone / iPad 可在 Safari 分享菜单选择“添加到主屏幕”；Android Chrome 可选择“安装应用”。更新会在后台下载，并在后续访问生效。

离线能力依赖用户至少成功访问过一次站点。`file://` 直接打开构建文件不支持 Service Worker，请通过 GitHub Pages 或本地 HTTP 服务访问。

## 文档评论

课程页底部集成了基于 GitHub Issues 的评论区。每个文档路径对应一个 Issue，读者使用 GitHub 账号发表评论；评论可以在仓库 Issues 中检索，也方便后续由 Agent 汇总并转化为文档改进。

首次启用需要完成一次仓库授权：

1. 确认 GitHub 仓库为公开仓库，并在仓库设置中启用 Issues。
2. 打开 [Utterances GitHub App](https://github.com/apps/utterances) 并点击 **Install**。
3. 选择 **Only select repositories**，授权 `glant509/ai-study`。
4. 等待 GitHub Pages 重新部署，然后打开任意课程页测试评论。

默认评论仓库是 `glant509/ai-study`。如需迁移，在构建环境中设置 `VITEPRESS_COMMENT_REPO=所有者/仓库名`。单篇文档可在 Frontmatter 中设置 `comments: false` 关闭评论。

维护者或 Agent 可按 `文档评论` 标签筛选 Issue。处理完成后，建议在评论中回复对应修改的提交，并关闭该 Issue；页面出现新评论时 Utterances 会重新打开它。

## 常用配置

- 站点与侧边栏：`docs/.vitepress/config.mts`
- 视觉样式：`docs/.vitepress/theme/style.css`
- PWA 图标：`docs/public/icon.svg`
- 自动部署：`.github/workflows/deploy.yml`

## License

课程内容和代码默认仅供个人学习与修改；公开发布前可按你的需要补充正式许可证。
