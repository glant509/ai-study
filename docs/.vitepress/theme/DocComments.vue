<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useData, useRoute } from 'vitepress'

const route = useRoute()
const { frontmatter, isDark, theme } = useData()
const container = ref<HTMLElement | null>(null)

const commentsEnabled = () =>
  theme.value.commentsEnabled === true &&
  frontmatter.value.layout !== 'home' &&
  frontmatter.value.comments !== false

function renderComments() {
  if (!container.value || !commentsEnabled()) return

  container.value.replaceChildren()
  const script = document.createElement('script')
  script.src = 'https://utteranc.es/client.js'
  script.async = true
  script.crossOrigin = 'anonymous'
  script.setAttribute('repo', theme.value.commentRepo)
  script.setAttribute('issue-term', 'pathname')
  script.setAttribute('label', '文档评论')
  script.setAttribute('theme', isDark.value ? 'github-dark' : 'github-light')
  container.value.appendChild(script)
}

function syncTheme(dark: boolean) {
  const iframe = container.value?.querySelector<HTMLIFrameElement>('.utterances-frame')
  iframe?.contentWindow?.postMessage(
    { type: 'set-theme', theme: dark ? 'github-dark' : 'github-light' },
    'https://utteranc.es'
  )
}

onMounted(renderComments)
watch(() => route.path, async () => {
  await nextTick()
  renderComments()
})
watch(isDark, syncTheme)
onBeforeUnmount(() => container.value?.replaceChildren())
</script>

<template>
  <section v-if="commentsEnabled()" class="doc-comments" aria-labelledby="doc-comments-title">
    <div class="doc-comments__heading">
      <div>
        <p class="doc-comments__eyebrow">交流与勘误</p>
        <h2 id="doc-comments-title">评论这篇文档</h2>
      </div>
      <span class="doc-comments__badge">GitHub</span>
    </div>
    <p class="doc-comments__intro">
      欢迎提出疑问、补充案例或标记需要完善的段落。评论会保存到仓库 Issue，供维护者和 Agent 后续处理。
    </p>
    <div ref="container" class="doc-comments__thread" />
    <noscript>需要启用 JavaScript 才能加载评论区。</noscript>
  </section>
</template>
