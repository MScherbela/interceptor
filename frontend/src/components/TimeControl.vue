<template>
  <div>
    <b-card>
      <b-row>
        <b-col>
          <b-time v-model="time" show-seconds locale="de" style="width:100%"/>
        </b-col>
      </b-row>
      <b-row>
        <b-col>
          <b-button-group class="mt-2" style="width:100%">
            <template v-for="speed in [0, 1, 10, 100]">
              <b-button size="sm" v-on:click="$store.commit('set_play_speed', speed)"
                        :variant="$store.state.play_speed == speed ? 'primary': ''" :key="speed">
                <b-icon-pause v-if="speed == 0"/>
                <b-icon-play v-else-if="speed == 1"/>
                <span v-else>{{ speed }}x</span>
              </b-button>
            </template>
          </b-button-group>
        </b-col>
      </b-row>
    </b-card>
  </div>
</template>

<script>
import {seconds_to_timestamp, timestamp_to_seconds} from "@/main";

export default {
  name: "TimeControl",
  computed: {
    time: {
      get() {
        return seconds_to_timestamp(this.$store.state.time)
      },
      set(t) {
        if (t.length > 0) {
          const new_time = timestamp_to_seconds(t)
          let delta_t = (new_time - this.$store.state.time)
          while (delta_t > 43200) {
            delta_t -= 86400
          }
          while (delta_t < -43200) {
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