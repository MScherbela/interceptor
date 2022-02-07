<template>
  <div>
    <b-card class="cardbackground">
      <b-row>
        <b-col>
          <b-time v-model="time" show-seconds locale="de" style="width:100%"/>
          <b-form-checkbox v-model="update_time" name="check-button" switch class="my-1">Karte aktualisieren</b-form-checkbox>

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
  data() {
    return {
      update_time: true
    }
  },
  computed: {
    time: {
      get() {
        return seconds_to_timestamp(this.$store.state.time)
      },
      set(t) {
        if (t.length > 0) {
          const new_time = timestamp_to_seconds(t)
          if (this.update_time) {
            let delta_t = (new_time - this.$store.state.time)
            while (delta_t > 43200) {
              delta_t -= 86400
            }
            while (delta_t < -43200) {
              delta_t += 86400
            }
            this.$store.commit('pass_time', delta_t)
          } else {
            this.$store.commit('set_time', new_time)
          }

        }
      }
    }
  },
}
</script>

<style scoped>

</style>