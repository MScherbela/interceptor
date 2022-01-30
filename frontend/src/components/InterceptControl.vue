<template>
  <b-card title="Abfangkurs">
    <b-button v-on:click="get_intercept_course">Berechnen</b-button>
    {{ a }}
  </b-card>
</template>


<script>
import axios from 'axios';

export default {
  name: "InterceptControl",
  data() {
    return {
      a: 0
    }
  },
  props: {
    ships: Array,
    uboot_pos: Object,
    time: Number
  },
  methods: {
    get_intercept_params() {
      const target = this.ships[0].pos
      return {
        x: this.uboot_pos.x,
        y: this.uboot_pos.y,
        heading: this.uboot_pos.heading,
        speed: this.uboot_pos.speed,
        target_x: target.x,
        target_y: target.y,
        target_heading: target.heading,
        target_speed: target.speed,
        time: this.time,
        n_segments: 2
      }
    },
    get_intercept_course() {
      axios.get("http://localhost:5000/api",
          {params: this.get_intercept_params()}
      ).then(response => {console.log(response.data); this.$store.commit('set_intercept', response.data)}
      ).catch(e => console.log(e))
    }
  }
}
</script>

<style scoped>

</style>