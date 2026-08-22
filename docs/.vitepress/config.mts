import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import { withPwa } from '@vite-pwa/vitepress'

const repoName = process.env.VITEPRESS_REPO_NAME ?? 'ai-agent-handbook'
const base = process.env.VITEPRESS_BASE ?? (repoName ? `/${repoName}/` : '/')

export default withPwa(withMermaid(defineConfig({
  lang: 'zh-CN',
  title: 'AI Agent 工程学习手册',
  description: '从 LLM 原理到 Agent Infrastructure 的工程化学习路径',
  base,
  cleanUrls: true,
  lastUpdated: true,
  head: [
    ['meta', { name: 'theme-color', content: '#0f766e' }],
    ['meta', { name: 'apple-mobile-web-app-capable', content: 'yes' }],
    ['meta', { name: 'apple-mobile-web-app-status-bar-style', content: 'default' }],
    ['link', { rel: 'icon', href: `${base}icon.svg`, type: 'image/svg+xml' }]
  ],
  markdown: {
    lineNumbers: true,
    theme: { light: 'github-light', dark: 'github-dark' }
  },
  themeConfig: {
    logo: '/icon.svg',
    siteTitle: 'AI Agent Handbook',
    search: { provider: 'local', options: { translations: {
      button: { buttonText: '搜索', buttonAriaLabel: '搜索文档' },
      modal: { noResultsText: '没有找到相关内容', resetButtonTitle: '清除查询', footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' } }
    } } },
    nav: [
      { text: '开始', link: '/guide/' },
      { text: '课程', link: '/llm/' },
      { text: '路线图', link: '/roadmap/' }
    ],
    sidebar: {
      '/': [
        { text: '开始学习', items: [
          { text: '手册导读', link: '/guide/' },
          { text: 'AI 工程全景', link: '/guide/overview' }
        ]},
        { text: '01 · LLM 基础', collapsed: false, items: [
          { text: '模块导读', link: '/llm/' },
          { text: 'Tokenizer', link: '/llm/tokenizer' },
          { text: 'Embedding 与张量', link: '/llm/embedding' },
          { text: 'Attention', link: '/llm/attention' },
          { text: 'Transformer', link: '/llm/transformer' },
          { text: 'RoPE', link: '/llm/rope' },
          { text: '推理与 KV Cache', link: '/llm/inference' }
        ]},
        { text: '02 · Agent Runtime', collapsed: false, items: [
          { text: '模块导读', link: '/agent/' },
          { text: 'Runtime 循环', link: '/agent/runtime' },
          { text: 'Tool Calling', link: '/agent/tools' },
          { text: 'Context Builder', link: '/agent/context-builder' },
          { text: 'Memory', link: '/agent/memory' },
          { text: 'Checkpoint', link: '/agent/checkpoint' },
          { text: 'Planner / Executor', link: '/agent/planner-executor' }
        ]},
        { text: '03 · RAG', collapsed: true, items: [
          { text: '模块导读', link: '/rag/' },
          { text: '检索流水线', link: '/rag/pipeline' },
          { text: '评测与优化', link: '/rag/evaluation' }
        ]},
        { text: '04 · MCP', collapsed: true, items: [
          { text: '协议与架构', link: '/mcp/' },
          { text: 'Server 实践', link: '/mcp/server' }
        ]},
        { text: '05 · Harness', collapsed: true, items: [
          { text: '运行容器', link: '/harness/' },
          { text: '多 Agent 协作', link: '/harness/multi-agent' }
        ]},
        { text: '06 · Infrastructure', collapsed: true, items: [
          { text: '推理服务', link: '/infra/' },
          { text: '可观测性', link: '/infra/observability' },
          { text: '扩展与调度', link: '/infra/scaling' }
        ]},
        { text: '07 · Security', collapsed: true, items: [
          { text: '威胁模型', link: '/security/' },
          { text: '权限与沙箱', link: '/security/permissions' }
        ]},
        { text: '08 · Roadmap', collapsed: true, items: [
          { text: '9 个月路线', link: '/roadmap/' },
          { text: '作品项目', link: '/roadmap/capstone' },
          { text: '学习检查表', link: '/roadmap/checklist' }
        ]}
      ]
    },
    outline: { level: [2, 3], label: '本页目录' },
    docFooter: { prev: '上一篇', next: '下一篇' },
    lastUpdated: { text: '最后更新' },
    darkModeSwitchLabel: '外观',
    sidebarMenuLabel: '目录',
    returnToTopLabel: '返回顶部',
    commentsEnabled: process.env.VITEPRESS_COMMENTS === 'true',
    commentRepo: process.env.VITEPRESS_COMMENT_REPO ?? 'glant509/ai-study'
  },
  sitemap: { hostname: process.env.VITEPRESS_SITE_URL ?? 'https://username.github.io/ai-agent-handbook/' },
  mermaid: { theme: 'base' }
}), {
  registerType: 'autoUpdate',
  includeAssets: ['icon.svg'],
  manifest: {
    name: 'AI Agent 工程学习手册',
    short_name: 'Agent 手册',
    description: '从 LLM 原理到 Agent Infrastructure 的工程化学习路径',
    theme_color: '#0f766e',
    background_color: '#f7f8f5',
    display: 'standalone',
    start_url: base,
    scope: base,
    lang: 'zh-CN',
    icons: [{ src: `${base}icon.svg`, sizes: 'any', type: 'image/svg+xml', purpose: 'any maskable' }]
  },
  workbox: {
    globPatterns: ['**/*.{css,js,html,svg,woff2}'],
    navigateFallback: `${base}index.html`,
    cleanupOutdatedCaches: true
  }
}))
