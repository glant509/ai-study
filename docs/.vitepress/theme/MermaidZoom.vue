<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'

const open = ref(false)
const svg = ref('')
const scale = ref(1)
const offsetX = ref(0)
const offsetY = ref(0)
let dragging = false
let startX = 0
let startY = 0

const canvasStyle = () => ({
  transform: `translate(-50%, -50%) translate(${offsetX.value}px, ${offsetY.value}px) scale(${scale.value})`
})

function reset() {
  scale.value = 1
  offsetX.value = 0
  offsetY.value = 0
}

function changeScale(delta: number) {
  scale.value = Math.min(4, Math.max(0.5, Number((scale.value + delta).toFixed(2))))
}

function showDiagram(diagram: Element) {
  const source = diagram.querySelector('svg')
  if (!source) return
  svg.value = source.outerHTML
  reset()
  open.value = true
  document.body.classList.add('mermaid-zoom-open')
}

function close() {
  open.value = false
  dragging = false
  document.body.classList.remove('mermaid-zoom-open')
}

function handleDocumentClick(event: MouseEvent) {
  const target = event.target as Element | null
  const diagram = target?.closest('.vp-doc .mermaid')
  if (diagram) showDiagram(diagram)
}

function handleKeydown(event: KeyboardEvent) {
  if (event.key === 'Escape' && open.value) close()
}

function handleWheel(event: WheelEvent) {
  changeScale(event.deltaY < 0 ? 0.15 : -0.15)
}

function startDrag(event: PointerEvent) {
  dragging = true
  startX = event.clientX - offsetX.value
  startY = event.clientY - offsetY.value
  ;(event.currentTarget as HTMLElement).setPointerCapture(event.pointerId)
}

function drag(event: PointerEvent) {
  if (!dragging) return
  offsetX.value = event.clientX - startX
  offsetY.value = event.clientY - startY
}

function stopDrag() {
  dragging = false
}

onMounted(() => {
  document.addEventListener('click', handleDocumentClick)
  document.addEventListener('keydown', handleKeydown)
})

onBeforeUnmount(() => {
  document.removeEventListener('click', handleDocumentClick)
  document.removeEventListener('keydown', handleKeydown)
  document.body.classList.remove('mermaid-zoom-open')
})
</script>

<template>
  <Teleport to="body">
    <div v-if="open" class="mermaid-zoom" role="dialog" aria-modal="true" aria-label="流程图大图浏览">
      <div class="mermaid-zoom__toolbar">
        <button type="button" aria-label="缩小" title="缩小" @click="changeScale(-0.2)">−</button>
        <span>{{ Math.round(scale * 100) }}%</span>
        <button type="button" aria-label="放大" title="放大" @click="changeScale(0.2)">＋</button>
        <button type="button" class="mermaid-zoom__reset" @click="reset">复位</button>
        <button type="button" class="mermaid-zoom__close" aria-label="关闭" title="关闭" @click="close">×</button>
      </div>
      <div
        class="mermaid-zoom__viewport"
        :class="{ 'is-dragging': dragging }"
        @wheel.prevent="handleWheel"
        @pointerdown="startDrag"
        @pointermove="drag"
        @pointerup="stopDrag"
        @pointercancel="stopDrag"
      >
        <div class="mermaid-zoom__canvas" :style="canvasStyle()" v-html="svg" />
      </div>
      <p class="mermaid-zoom__hint">滚轮或按钮缩放 · 按住拖动浏览 · Esc 关闭</p>
    </div>
  </Teleport>
</template>
