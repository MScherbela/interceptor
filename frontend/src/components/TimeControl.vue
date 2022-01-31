<template>
  <div>
    <b-card>
      <b-time v-model="time" show-seconds locale="de"/>
      <b-button-group class="mt-2">
        <template v-for="speed in [0, 1, 10, 100]">
          <b-button size="sm" v-on:click="$store.commit('set_play_speed', speed)" :variant="$store.state.play_speed == speed ? 'primary': ''" :key="speed">
            <b-icon-pause v-if="speed == 0"/>
            <b-icon-play v-else-if="speed == 1"/>
            <span v-else>{{ speed }}x</span>
          </b-button>
        </template>
      </b-button-group>
    </b-card>
  </div>
</template>

<script>
export default {
  name: "TimeControl",
  computed: {
    time: {
      get() {
        let time = this.$store.state.time
        const h = Math.floor(time / 3600)
        const m = Math.floor(((time - 3600 * h) / 60))
        const s = Math.floor((time - 3600 * h - 60 * m))
        return h + ":" + m + ":" + s
      },
      set(t) {
        if (t.length > 0) {
          const h = parseFloat(t.substring(0, 2))
          const m = parseFloat(t.substring(3, 5))
          const s = parseFloat(t.substring(6, 8))
          const new_time = 3600 * h + 60 * m + s
          let delta_t = (new_time - this.$store.state.time)
          while(delta_t > 43200){
            delta_t -= 86400
          }
          while(delta_t < -43200){
            delta_t += 86400
          }
          this.$store.commit('pass_time', delta_t)
        }
      }
    }
  },
}
</script>

<style scoped>

</style>