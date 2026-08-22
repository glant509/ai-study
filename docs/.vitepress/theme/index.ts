import DefaultTheme from 'vitepress/theme'
import { h } from 'vue'
import DocComments from './DocComments.vue'
import './style.css'

export default {
  extends: DefaultTheme,
  Layout: () => h(DefaultTheme.Layout, null, {
    'doc-after': () => h(DocComments)
  })
}
